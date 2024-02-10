#!/usr/bin/env python3
import argparse
import copy
import json
import re
import sys

from abc import ABC, abstractmethod
from collections import UserDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from typing import Any, BinaryIO, Callable, cast, get_args, Generic, Generator, Literal, NamedTuple, Optional, overload, TypeVar, Type, Union


ROOT = Path(__file__).absolute().parent


class Preset(ABC):
    def __init__(self, data: Union[bytes, list[int]]) -> None:
        self._data = list(data) if isinstance(data, bytes) else data

    @property
    @abstractmethod
    def _offset(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def _name_range(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        s, e = self._name_range
        d = self[s:e]
        return bytearray(d).decode('ascii').strip()

    @overload
    def __getitem__(self, sub: int) -> int:
        ...

    @overload
    def __getitem__(self, sub: slice) -> list[int]:
        ...

    def __getitem__(self, sub: Union[int, slice]) -> Union[int, list[int]]:
        if isinstance(sub, slice):
            return self._data[self._offset+sub.start:self._offset+sub.stop:sub.step]
        return self._data[self._offset+sub]

    def __setitem__(self, idx: int, val: int) -> None:
        self._data[self._offset+idx] = val

    def valid(self) -> bool:
        return self._data[0] == 240 and self._data[-1] == 247

    def bytes(self) -> bytes:
        return bytearray(self._data)


class CircuitPreset(Preset):
    _name_range = (0, 16)

    @property
    def _offset(self) -> int:
        return 9 if self._data[6] == 0 else 11

    def valid(self) -> bool:
        sysex = super(CircuitPreset, self).valid()
        return sysex and self._data[1:5] == [0, 32, 41, 1]


class UltraNovaPreset(Preset):
    _name_range = (0, 16)
    _offset = 15

    def valid(self) -> bool:
        sysex = super(UltraNovaPreset, self).valid()
        return sysex and self._data[1:6] == [0, 32, 41, 3, 1]


@dataclass
class ParamTemplate:
    offset: int
    default: int
    min: int
    max: int


@dataclass
class Param(ParamTemplate):
    _value: int

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        self._value = self.limit(v)

    def validate(self, value: int, key:str ='param') -> None:
        if value < self.min:
            raise ValueError(f'{key} value {value} < {self.min}')
        if value > self.max:
            raise ValueError(f'{key} value {value} > {self.max}')

    def limit(self, val: int) -> int:
        val = self.min if val < self.min else val
        val = self.max if val > self.max else val
        return val


class ParamsTemplate(UserDict[str, ParamTemplate]):
    @property
    def initial(self) -> dict[str, Param]:
        return {
            k:Param(
                offset=v.offset,
                default=v.default,
                min=v.min,
                max=v.max,
                _value=0
            )
            for k, v in self.data.items()
        }


class ParamsTemplateDecoder(json.JSONDecoder):
    def __init__(self) -> None:
        json.JSONDecoder.__init__(self, object_hook=self.object_hook)

    def object_hook(self, dct: dict[str, Any]) -> Union[ParamsTemplate, ParamTemplate]:
        if 'offset' in dct and 'default' in dct and 'min' in dct and 'max' in dct:
            return ParamTemplate(
                offset=dct['offset'],
                default=dct['default'],
                min=dct['min'],
                max=dct['max']
            )
        if all([isinstance(v, ParamTemplate) for v in dct.values()]):
            return ParamsTemplate(dct)
        raise Exception(f'invalid params template {dct}')


def load_params_template(path: Path) -> ParamsTemplate:
    with open(path, 'r') as fp:
        return ParamsTemplate(json.load(fp, cls=ParamsTemplateDecoder))


class Params(UserDict[str, Param], ABC):
    @property
    @abstractmethod
    def _params(self) -> ParamsTemplate:
        raise NotImplementedError

    @property
    @abstractmethod
    def _template(self) -> Preset:
        raise NotImplementedError

    @property
    def strict(self) -> bool:
        return self._strict

    def __init__(self, preset: Optional[Preset]=None, strict: bool=True) -> None:
        self._strict = strict
        data = preset if preset is not None else self._template
        params = self._params.initial
        for key, param in params.items():
            value = data[param.offset]
            if self.strict:
                param.validate(value, key)
            param.value = value
        super(Params, self).__init__(params)

    def dump(self) -> Preset:
        data = copy.deepcopy(self._template)
        for key, param in self.items():
            value = param.value
            if self.strict:
                param.validate(value, key)
            data[param.offset] = value
        return data


@dataclass
class NovaModulation:
    idx: int
    src1: Param
    src2: Param
    depth: Param
    dst: Param


CircuitModulation = NovaModulation


CircuitMacroParams = Literal['A', 'B', 'C', 'D']

@dataclass
class CircuitMacro:
    idx: int
    param: CircuitMacroParams
    pos: Param
    dst: Param
    dstkey: str
    start: Param
    end: Param
    depth: Param


class CircuitParams(Params):
    _params = load_params_template(ROOT / 'circuit_params.json')

    RE_MACRO_DST = re.compile('MacroKnob(\\d+)_Destination([A-D])')

    MACRO_DST_MAP = [
        '---',
        'Voice_PortamentoRate',
        'Mixer_PostFXLevel',
        'Osc1_WaveInterpolate', 'Osc1_PulseWidthIndex', 'Osc1_VirtualSyncDepth',
        'Osc1_Density', 'Osc1_DensityDetune', 'Osc1_Semitones', 'Osc1_Cents',
        'Osc2_WaveInterpolate', 'Osc2_PulseWidthIndex', 'Osc2_VirtualSyncDepth',
        'Osc2_Density', 'Osc2_DensityDetune', 'Osc2_Semitones', 'Osc2_Cents',
        'Mixer_Osc1Level', 'Mixer_Osc2Level', 'Mixer_RingModLevel12', 'Mixer_NoiseLevel',
        'Filter1_Frequency', 'Filter1_Resonance', 'Filter1_Drive', 'Filter1_Track', 'Filter1_Env2ToFreq',
        'Envelope1_Attack', 'Envelope1_Decay', 'Envelope1_Sustain', 'Envelope1_Release',
        'Envelope2_Attack', 'Envelope2_Decay', 'Envelope2_Sustain', 'Envelope2_Release',
        'Envelope3_Delay',
        'Envelope3_Attack', 'Envelope3_Decay', 'Envelope3_Sustain', 'Envelope3_Release',
        'LFO1_Rate', 'LFO1_PhaseOffset', 'LFO1_SlewRate',
        'LFO2_Rate', 'LFO2_PhaseOffset', 'LFO2_SlewRate',
        'Distortion1_Level',
        'Chorus1_Level', 'Chorus1_Rate', 'Chorus1_Feedback', 'Chorus1_ModDepth', 'Chorus1_Delay',
        'ModMatrix1_Depth', 'ModMatrix2_Depth', 'ModMatrix3_Depth', 'ModMatrix4_Depth',
        'ModMatrix5_Depth', 'ModMatrix6_Depth', 'ModMatrix7_Depth', 'ModMatrix8_Depth',
        'ModMatrix9_Depth', 'ModMatrix10_Depth', 'ModMatrix11_Depth', 'ModMatrix12_Depth',
        'ModMatrix13_Depth', 'ModMatrix14_Depth', 'ModMatrix15_Depth', 'ModMatrix16_Depth',
        'ModMatrix17_Depth', 'ModMatrix18_Depth', 'ModMatrix19_Depth', 'ModMatrix20_Depth',
    ]

    @property
    def _template(self) -> Preset:
        raise NotImplementedError

    @property
    def modulations(self) -> Generator[CircuitModulation, None, None]:
        params = self.data
        for i in range(1, 21):
            yield CircuitModulation(
                idx=i,
                src1=params[f'ModMatrix{i}_Source1'],
                src2=params[f'ModMatrix{i}_Source2'],
                depth=params[f'ModMatrix{i}_Depth'],
                dst=params[f'ModMatrix{i}_Destination']
            )

    @property
    def macros(self) -> Generator[CircuitMacro, None, None]:
        params = self.data
        for i in range(1, 9):
            pos = params[f'MacroKnob{i}_Position']
            for p in get_args(CircuitMacroParams):
                dst = params[f'MacroKnob{i}_Destination{p}']
                if dst.value < 0 or dst.value >= len(CircuitParams.MACRO_DST_MAP):
                    raise Exception(f'invalid macro destination {i}{p}')
                yield CircuitMacro(
                        idx=i,
                        pos=pos,
                        param=p,
                        dst=dst,
                        dstkey = CircuitParams.MACRO_DST_MAP[dst.value],
                        start=params[f'MacroKnob{i}_StartPos{p}'],
                        end=params[f'MacroKnob{i}_EndPos{p}'],
                        depth=params[f'MacroKnob{i}_Depth{p}'],
                )

    def apply_macros(self) -> None:
        # apply macro values to params
        for macro in self.macros:
            pos = macro.pos.value
            dst = macro.dst.value
            if pos <= 0 or dst <= 0:
                continue
            start = macro.start.value
            end = macro.end.value
            depth = macro.depth.value
            # compute mod value
            depth = depth - 64 if depth >= 64 else -(64 - depth)
            delta = end - start
            if delta == 0:
                rval = 0.0 if pos == start else 1.0
            else:
                rval = (min(max(pos, start), end) - start) / delta
            mod = int(rval*depth*2)
            # get destination param
            #print(f'{macro.idx} {macro.param} -> {macro.dstkey} -> {pos} {depth} {rval} {mod}')
            dstparam = self.data[macro.dstkey]
            # compute new val
            val = dstparam.value
            val += mod
            dstparam.value = val
            # set depth to default (zero) as macro modulation has been applied
            macro.depth.value = macro.depth.default


@dataclass
class UltraNovaModulation(NovaModulation):
    trigger: Param


class UltraNovaParams(Params):
    _template = UltraNovaPreset([240, 0, 32, 41, 3, 1, 127, 0, 0, 16, 0, 0, 0, 9, 7, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0, 0, 0, 2, 0, 64, 0, 0, 0, 25, 65, 0, 0, 1, 0, 2, 127, 64, 0, 127, 0, 0, 64, 64, 76, 2, 127, 64, 0, 127, 0, 0, 64, 64, 76, 2, 127, 64, 0, 127, 0, 0, 64, 64, 76, 127, 0, 0, 0, 0, 0, 0, 64, 64, 127, 3, 0, 0, 0, 0, 3, 127, 127, 0, 64, 64, 0, 0, 3, 127, 127, 0, 64, 64, 64, 40, 0, 0, 64, 0, 0, 63, 60, 0, 64, 2, 90, 127, 40, 64, 127, 0, 64, 64, 64, 0, 127, 0, 64, 2, 75, 35, 45, 64, 127, 0, 64, 64, 64, 0, 127, 0, 0, 10, 70, 64, 40, 64, 127, 0, 64, 64, 64, 0, 127, 0, 0, 10, 70, 64, 40, 64, 127, 0, 64, 64, 64, 0, 127, 0, 0, 10, 70, 64, 40, 64, 127, 0, 64, 64, 64, 0, 127, 0, 0, 10, 70, 64, 40, 64, 127, 0, 64, 64, 64, 0, 127, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 4, 0, 10, 1, 4, 6, 8, 0, 127, 0, 0, 0, 64, 64, 64, 64, 64, 64, 10, 44, 0, 64, 32, 127, 10, 44, 0, 64, 32, 127, 0, 100, 64, 0, 100, 64, 64, 9, 64, 127, 6, 0, 64, 0, 64, 127, 0, 0, 4, 90, 64, 4, 90, 64, 1, 20, 0, 74, 64, 64, 1, 20, 0, 74, 64, 64, 1, 20, 0, 74, 64, 64, 1, 20, 0, 74, 64, 64, 3, 7, 0, 127, 16, 64, 64, 0, 64, 127, 40, 64, 64, 0, 0, 3, 64, 0, 0, 0, 0, 0, 0, 126, 0, 0, 20, 64, 36, 80, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 63, 7, 7, 56, 7, 7, 7, 7, 7, 0, 63, 7, 63, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 247])
    _params = load_params_template(ROOT / 'ultranova_params.json')

    @property
    def modulations(self) -> Generator[UltraNovaModulation, None, None]:
        params = self.data
        for i in range(1, 21):
            yield UltraNovaModulation(
                idx=i,
                src1=params[f'ModMatrix{i}_Source1'],
                src2=params[f'ModMatrix{i}_Source2'],
                trigger=params[f'ModMatrix{i}_TouchTrigger'],
                depth=params[f'ModMatrix{i}_Depth'],
                dst=params[f'ModMatrix{i}_Destination']
            )


SourceParams = TypeVar('SourceParams', bound=Params)
DestParams = TypeVar('DestParams', bound=Params)


class ParamsTranslator(ABC, Generic[SourceParams, DestParams]):
    @abstractmethod
    def _remap_key(self, key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def _remap_param(self, key: str, src: Param, dst: Param) -> None:
        raise NotImplementedError

    @abstractmethod
    def _finalize(self, source: SourceParams, destination: DestParams) -> None:
        raise NotImplementedError

    @abstractmethod
    def _unmapped_key(self, key: str) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def _destination(self) -> Type[DestParams]:
        raise NotImplementedError

    def translate(self, source: SourceParams) -> DestParams:
        destination = self._destination()
        for key, srcparam in [(self._remap_key(p), m) for p, m in source.items()]:
            # sync value
            if key in destination:
                self._remap_param(key, srcparam, destination[key])
            else:
                self._unmapped_key(key)
        self._finalize(source, destination)
        return destination


class CircuitToNova(ParamsTranslator[CircuitParams, UltraNovaParams]):
    _destination = UltraNovaParams

    FILTER_MAP_C2V = {
        0:1,  1:3,  2:4,  # LP12, LP24, BP6
        3:5, 4:11, 5:13   # BP 12, HP12, HP24
    }

    SRC_MAP_C2V = {
        0:0,   4:4,   5:5,  # Direct, Velocity, Keyboard
        6:6,   7:7,   8:8,  # LFO1+, LFO1+/-, LFO2+
        9:9, 10:12, 11:13,  # LFO2+/-, Env1, Env2
        12:14               # Env3
    }

    DST_MAP_C2V = {
        # Osc2 is remapped to Osc3 in all parameters
        0:0,     1:1,   2:3,  # Osc1/2 Pitch, Osc1 Pitch, Osc2 Pitch
        3:4,     4:6,   5:7,  # Osc1 VSync, Osc2 VSync, Osc1 PW
        6:9,    7:13,  8:15,  # Osc2 PW, Osc1 Level, Osc2 Level
        9:16,  10:17, 11:19,  # Noise Level, Ring Mod 1/2 Level, Filter Drive
        12:21, 13:23, 14:26,  # Filter Freq, Filter Reso, LFO1 Rate
        15:27, 16:29, 17:30   # LFO2 Rate, Env1 Decay, Env2 Decay
    }

    WAV_MAP_C2V = {
        0:0,     1:1,   2:2,  # Sine, Triangle, Saw
        3:3,     4:4,   5:5,  # Saw [9:1] PW, Saw [8:2] PW, Saw [7:3] PW
        6:6,     7:7,   8:8,  # Saw [6:4] PW, Saw [5:5] PW, Saw [4:6] PW
        9:9,   10:10, 11:11,  # Saw [3:7] PW, Saw [2:8] PW, Saw [1:9] PW
        12:12, 13:13,         # PW, Square
        # wave tables are very different between Circuit and UltraNova, remapping to something 'close enough'
        14:48, 15:36,         # Sine Table -> WT15, Analogue Pulse -> WT3
        16:41, 17:47,         # Analogue Sync -> WT8, Tri-Saw Blend -> WT14
        18:43, 19:44,         # Nasty 1 -> WT10, Nasty 2 -> WT11
        20:42, 21:68,         # SSQ -> WT9, Vocal 1 -> WT35
        22:62, 23:58,         # Vocal 2 -> WT29, Vocal 3 -> WT25
        24:55, 25:53,         # Vocal 4 -> WT22, Vocal 5 -> WT20
        26:52, 27:36,         # Vocal 6 -> WT19, Collection 1 -> WT3
        28:34, 29:48          # Collection 2 -> WT1, Collection 3 -> WT15
    }

    # map Circuit macros' destinations to UltraNova tweaks' assignments
    MACRO_MAP_C2V = {
        0:0,       1:1,    2:3,
        3:6,       4:7,    5:8,
        6:10,     7:11,   8:12,   9:13,
        10:22,   11:23,  12:24,
        13:26,   14:27,  15:28,  16:29,
        17:30,   18:32,  19:33,  20:35,
        21:37,   22:38,  23:39,  24:40,  25:45,
        26:47,   27:48,  28:49,  29:50,
        30:51,   31:52,  32:53,  33:54,
        34:55,
        35:56,   36:57,  37:58,  38:59,
        39:60,   40:61,  41:62,
        42:63,   43:64,  44:65,
        45:71,
        46:69,   47:85,  48:86,  49:87,  50:88,
        51:106, 52:107, 53:108, 54:109,
        55:110, 56:111, 57:112, 58:113,
        59:114, 60:115, 61:116, 62:117,
        63:118, 64:119, 65:120, 66:121,
        67:122, 68:123, 69:124, 70:125
    }

    def _remap_key(self, key: str) -> str:
        # handle ring mod by remapping Osc2 to Osc3
        if key.startswith('Osc2_'):
            key = 'Osc3_' + key[5:]
        elif key == 'Mixer_Osc2Level':
            key = 'Mixer_Osc3Level'
        elif key == 'Mixer_RingModLevel12':
            key = 'Mixer_RingModLevel13'
        # handle FX based on template FX chain (Chorus -> EQ -> Dist -> Delay -> Reverb)
        elif key == 'Chorus1_Level':
            key = 'FX1_Level'
        elif key == 'Distortion1_Level':
            key = 'FX3_Level'
        else:
            m = CircuitParams.RE_MACRO_DST.match(key)
            if m is not None:
                idx = int(m.group(1))
                par = m.group(2)
                if par == 'A':
                    key = f'Tweak{idx}_Assignment'
        return key

    def _remap_param(self, key: str, src: Param, dst: Param) -> None:
        val = src.value
        if key == 'Voice_KeyboardOctave':
            if val < 64:
                val = 128 - (64 - val)
            else:
                val = val - 64
        elif key.startswith('Osc') and key.endswith('_Wave'):
            val = CircuitToNova.WAV_MAP_C2V.get(val, dst.default)
        elif key == 'Filter_Routing':
            if val == 0:  # no bypass
                val = 1  # single
            elif val == 2:  # bypass Osc1&2
                val = 0  # bypass
            elif val == 1:  # bypass Osc1
                val = 4  # parallel 2 (Osc1 is routed to fully open LP filter)
        elif key == 'Filter1_Type':
            val = CircuitToNova.FILTER_MAP_C2V.get(val, dst.default)
        elif 'ModMatrix' in key and 'Source' in key:
            val = CircuitToNova.SRC_MAP_C2V.get(val, dst.default)
        elif 'ModMatrix' in key and 'Destination' in key:
            val = CircuitToNova.DST_MAP_C2V.get(val, dst.default)
        elif key.startswith('Tweak') and key.endswith('_Assignment'):
            val = CircuitToNova.MACRO_MAP_C2V.get(val, dst.default)
        elif key == 'Mixer_PostFXLevel':
            val = dst.limit(val)  # UltraNova is limited to +12db
        dst.validate(val, key)
        dst.value = val

    def _finalize(self, source: CircuitParams, destination: UltraNovaParams) -> None:
        # setup parallel filter
        if destination['Filter_Routing'].value == 4:
            destination['Filter_Balance'].value = 64
            destination['Filter2_Type'].value = 0
        # map modwheel to filter freq (if any macro was assigned to it)
        has_filter_macro = any([True for macro in source.macros if macro.dstkey == 'Filter1_Frequency'])
        if has_filter_macro:
            for mod in destination.modulations:
                if mod.src1.value != 0 or mod.src2.value != 0:
                    continue
                mod.src1.value = 1 # modwheel
                mod.dst.value = 21 # filter 1 freq
                mod.depth.value = 127
                break

    def _unmapped_key(self, key: str) -> None:
        if not key.startswith('MacroKnob'):
            raise Exception(f'unmapped param {key}')


def circuit_to_nova(source: BinaryIO, destination: BinaryIO) -> str:
    circuit_syx = CircuitPreset(source.read())
    if not circuit_syx.valid():
        raise Exception('input is not a valid preset file!')

    circuit_params = CircuitParams(circuit_syx)
    circuit_params.apply_macros()

    converter = CircuitToNova()

    ultranova_params = converter.translate(circuit_params)
    ultranova_syx = ultranova_params.dump()

    destination.write(ultranova_syx.bytes())

    return circuit_syx.name


def DirType(mode: str) -> Callable[[str], Path]:
    def dir_arg(arg: str) -> Path:
        path = Path(arg)
        if path.exists():
            if not path.is_dir():
                raise argparse.ArgumentTypeError(f'{path} is not a dir')
        elif mode == 'r':
            raise argparse.ArgumentTypeError(f'invalid path {path}')
        elif mode == 'w':
            path.mkdir()
        return path
    return dir_arg


def name_variant(base: Path, name: str, variant: int=65) -> Path:
    prefix = ' ' + chr(variant) if variant > 65 else ''
    res = base.joinpath(name+prefix+'.syx')
    return res if not res.exists() else name_variant(base, name, variant+1)


class CliMode(Enum):
    SINGLE = 1
    BATCH = 2


class CliArgs(NamedTuple):
    mode: CliMode
    infile: BinaryIO
    outfile: BinaryIO
    indir: Path
    outdir: Path
    rename: bool


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert Circuit Tracks presets to UltraNova')
    subparsers = parser.add_subparsers(required=True)
    parser_single = subparsers.add_parser('single', help='convert a single file')
    parser_single.set_defaults(mode=CliMode.SINGLE)
    parser_single.add_argument('infile', metavar='circuit.syx', type=argparse.FileType('rb'))
    parser_single.add_argument('outfile', metavar='ultranova.syx', type=argparse.FileType('wb'))
    parser_batch = subparsers.add_parser('batch', help='convert multiple files')
    parser_batch.set_defaults(mode=CliMode.BATCH)
    parser_batch.add_argument('indir', metavar='INPUTDIR', type=DirType('r'))
    parser_batch.add_argument('outdir', metavar='OUTDIR', type=DirType('w'))
    parser_batch.add_argument('-r', '--rename', action='store_true', help='rename created syx files based on preset name')

    args = cast(CliArgs, parser.parse_args())

    if args.mode == CliMode.SINGLE:
        name = circuit_to_nova(args.infile, args.outfile)
        print(f'Preset "{name}" converted!')
    elif args.mode == CliMode.BATCH:
        if args.indir == args.outdir:
            sys.exit('input and output directories cannot be the same!')
        for idx, path in enumerate(sorted(args.indir.glob('*.syx'))):
            opath = args.outdir.joinpath(path.name)
            with open(path, 'rb') as fpin:
                with open(opath, 'wb') as fpout:
                    name = circuit_to_nova(fpin, fpout)
            if args.rename:
                target = name_variant(args.outdir, name)
                opath.rename(target)
            print(f'[{idx}] Preset "{name}" converted!')


if __name__ == '__main__':
    main()
