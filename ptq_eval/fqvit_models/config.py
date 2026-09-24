from .ptq.bit_type import BIT_TYPE_DICT


class Config:

    def __init__(self, ptf=None, lis=None, quant_method='minmax', pcs=None,
                 linear_only=False):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".

        pcs (added 2026-04-22, Andersen fork) = Per-Channel Scale. Continuous
        FP per-channel scale at LN input (minmax observer, channel_wise mode),
        no PoT rounding. LN body runs FP32 (INT_NORM=False) — isolates the
        per-channel granularity contribution from PTF's integer-domain bundle.
        Mutually exclusive with ptf.

        DEFAULT (2026-06-25): bare Config() => official FQ-ViT "888" = PTF+LIS, the
        config that REPRODUCES THE PAPER (see Config.fqvit_official(); e.g. swin_tiny
        ~80.2 vs README 80.51). PCS remains an explicit FQ-config opt-in via
        Config(pcs=True) for legacy equivalence studies. No typed APO token maps
        onto that native path: Nfpcs is Whisper-only APO, while Nipcs is integer. The
        sentinel fires ONLY on a truly bare Config() (ptf/lis/pcs all None); any
        explicit flag wins, so positional callers passing explicit False
        (test_quant.py:147 Config(args.ptf, args.lis, ...) with no CLI flags =>
        plain minmax baseline) and Config(pcs=True) are unaffected.

        History: the Andersen fork (2026-06-22 .. 2026-06-25) had bare Config() => PCS
        to make the default the apo-equivalence config; reverted because the repo
        default should reproduce the official paper. Explicit FQ PCS callers are
        unaffected by that default change.
        '''
        # Sentinel: a TRULY bare Config() (ptf/lis/pcs all unspecified) => official
        # FQ-ViT "888" = PTF+LIS (reproduces the paper; see fqvit_official()). Any
        # explicit flag -- incl. the positional False that test_quant.py passes when
        # no --ptf/--lis is given -- bypasses this and is honored as written.
        if ptf is None and lis is None and pcs is None and not linear_only:
            ptf = True
            lis = True
        ptf = bool(ptf)
        lis = bool(lis)
        pcs = bool(pcs)
        if ptf and pcs:
            raise ValueError("ptf and pcs are mutually exclusive LN schemes")
        if linear_only and (ptf or pcs):
            raise ValueError(
                "linear_only disables all non-linear-adjacent QActs; ptf/pcs (which "
                "reshape the LN-input QAct) are no-ops in this mode — set at most one."
            )
        self.LINEAR_ONLY = linear_only
        # Guard: PCS needs a continuous-scale observer (minmax / percentile / ...),
        # NOT the PoT-rounding 'ptf' observer — otherwise PCS semantically degrades
        # to PTF without the INT_NORM body. Mutex above only catches the ptf=True
        # flag; this catches `Config(pcs=True, quant_method='ptf')`.
        _PCS_ALLOWED_OBSERVERS = ('minmax', 'percentile', 'omse', 'ema')
        if pcs and quant_method not in _PCS_ALLOWED_OBSERVERS:
            raise ValueError(
                f"pcs=True requires quant_method in {_PCS_ALLOWED_OBSERVERS}, "
                f"got {quant_method!r} (would silently fall back to PoT-like rep)"
            )

        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        if lis and linear_only:
            raise ValueError("lis is incompatible with linear_only (SM op stays FP)")
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        elif pcs:
            # Per-channel continuous FP scale at LN input, FP LN body.
            # Shares only the channel_wise calibration granularity with PTF;
            # the *observer* differs: PTF uses 'ptf' (PoT mask via
            # PtfObserver.get_quantization_params), PCS uses a continuous
            # observer (typically 'minmax') so scales stay FP. INT_NORM=False
            # keeps QIntLayerNorm in mode='ln' (F.layer_norm body),
            # bypassing the integer-LN math trick entirely.
            self.INT_NORM = False
            self.OBSERVER_A_LN = quant_method          # guarded above; not 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A

    def quant_states(self):
        """Canonical L/N/S/G semantic states after ``model_quant()``.

        Q-DQ boundaries feeding the next Linear belong to the Linear axis in the
        ``linear_only`` attribution, so all SF axes are ``no`` in that mode.
        """
        if self.LINEAR_ONLY:
            return {"linear": "fake", "ln": "no", "sm": "no", "gelu": "no"}
        return {"linear": "fake",
                "ln": "int" if self.INT_NORM else "fake",
                "sm": "int" if self.INT_SOFTMAX else "fake",
                "gelu": "fake"}

    @classmethod
    def fqvit_official(cls, quant_method='minmax'):
        """Canonical config that reproduces the official FQ-ViT paper ("Ours" 8/8/8):
        Power-of-Two Factor integer LayerNorm (PTF) + Log-Int-Softmax (LIS). This is
        the SAME thing a bare ``Config()`` now resolves to -- the named factory just
        makes the intent explicit and self-documenting at call sites. Equivalent CLI:
        ``test_quant.py <model> <data> --quant --ptf --lis --quant-method minmax``
        (reproduces e.g. swin_tiny ~80.2 vs README 80.51)."""
        return cls(ptf=True, lis=True, quant_method=quant_method)
