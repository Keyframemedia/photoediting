"""Signature looks. Values are tuned on real Keyframe brackets; see README."""

DAY = {
    "name": "day",
    # Fitted to Keyframe's delivered day edits (Water Lily, Foxglove): each bracket
    # rendered and aligned to the delivered JPEG, parameters searched to minimise the
    # Oklab difference, then the house LUT fitted on the remainder (luts/day.npy).
    # exposure: scene median (windows/sky excluded) to `key`, plus bias
    "exposure_mode": "median", "key": 0.33, "key_pct": 50, "key_exclude_top": 0.10,
    "exposure_bias": 0.0,
    # white balance: neutral surfaces land on a warm-clean white (Planckian target)
    "wb_target_cct": 5600, "wb_strength": 0.75, "wb_tint": 0.0, "wb_max_mired": 45,
    "wb_max_warm_mired": 12,
    # local tone mapping: hat-weighted fusion of virtual exposures
    "fusion_mode": "hat", "fusion_evs": [-6.0, -4.5, -3.0, -1.5, 0.0, 1.0],
    "fusion_hat": [0.0, 0.06, 0.96, 1.0], "fusion_ev_prior": 0.7,
    "fusion_sigma": 0.22,
    # skies/views: one global exposure for natural clouds and a clean window view.
    # A narrow transition (hi 2.4) and near-white kept in the fusion (hat .96) leave
    # ceilings beside windows evenly bright; wider settings put a grey halo around
    # every window (and match the delivered set less well).
    "sky_global": 1.0, "sky_global_lo": 1.0, "sky_global_hi": 2.4,
    "sky_global_pct": 50.0, "sky_global_white": 0.60, "sky_global_shoulder": 4.0,
    # levels + tone: bright and open, gentle contrast
    "levels": True, "black_pct": 0.05, "white_pct": 99.5, "white_target": 0.99,
    "max_white_stretch": 1.3,
    "black": 0.0, "white": 1.0, "contrast": 0.15, "pivot": 0.25, "toe": 0.04, "shoulder": 0.06,
    "clarity": 0.10, "clarity_sigma_frac": 0.012, "micro_contrast": 0.20, "micro_sigma_frac": 0.002,
    "desat_highlights": 0.7,
    # colour (Oklab units); most of the house colour is in the LUT
    "vibrance": 0.10, "saturation": 0.0, "warmth_b": 0.006, "tint_a": 0.0,
    "sky_sat": 0.0, "sky_deepen": 0.015, "green_warm_deg": -6.0, "green_sat": 0.05,
    "orange_protect": 0.5,
    "lut": "day", "lut_strength": 1.0,
    # sharpening
    "sharpen_radius": 0.9, "sharpen_amount": 0.55, "web_sharpen_radius": 0.6,
    "web_sharpen_amount": 0.55, "sharpen_threshold": 0.006,
}

# Night: a moody warm interior glow against a deep blue sky. White balance is
# fixed (not auto) so tungsten/LED interiors stay golden and the dusk sky blue.
NIGHT = dict(DAY)
NIGHT.update({
    "name": "night",
    "key": 0.16, "key_exclude_top": 0.15,
    "wb_source": "daylight", "wb_fixed_cct": 4100, "wb_target_cct": 6300,
    "fusion_evs": [-8.0, -6.5, -5.0, -3.5, -2.0, -1.0, 0.0, 0.8],
    "fusion_hat": [0.0, 0.05, 0.86, 0.99], "fusion_ev_prior": 0.45,
    "black_pct": 0.5, "white_pct": 99.8, "contrast": 0.36, "toe": 0.05,
    "clarity": 0.28, "micro_contrast": 0.15,
    "vibrance": 0.30, "saturation": 0.10, "warmth_b": 0.004,
    "sky_sat": 0.22, "sky_deepen": 0.06, "green_sat": 0.0,
    "chroma_nr": 1.6, "luma_nr": 0.6,
    "desat_highlights": 0.45,
    # every interior/exterior light reads as on (lights.py)
    "lights": True, "lights_glow": 0.35, "lights_pool": 0.18, "lights_warmth": 0.04,
    # held from the earlier day look this was built on
    "sky_global": 0.9, "sky_global_hi": 2.4, "sky_global_white": 0.6, "white_target": 0.96,
    "max_white_stretch": 1.12, "pivot": 0.42, "shoulder": 0.06, "lut": None,
})

# Twilight: Keyframe's delivered dusk look, fitted on Water Lily (dusk) the same way
# as DAY, with the sky masked out: bright facades, warm timber and glowing windows.
# With a replaced sky, the house gradient (skydome.GRADIENTS) is painted over it.
TWILIGHT = dict(NIGHT)
TWILIGHT.update({
    "name": "twilight",
    "key": 0.25, "wb_fixed_cct": 8400, "wb_target_cct": 5200,
    "fusion_ev_prior": 0.45, "contrast": 0.36, "pivot": 0.42, "black_pct": 0.5,
    "sky_global": 0.5, "sky_global_white": 0.9, "max_white_stretch": 1.12, "white_target": 0.99,
    "vibrance": 0.30, "saturation": 0.10, "warmth_b": 0.012, "tint_a": 0.008,
    "sky_sat": 0.0, "sky_deepen": 0.06, "clarity": 0.10, "desat_highlights": 0.45,
    "sky_wisps": 0.8,
    # interiors (pipeline.finish decides): lamps and LEDs kept warm-cream, not
    # orange, with the dusk blue left in the windows
    "interior": {"wb_fixed_cct": 4200, "warmth_b": 0.006, "key": 0.29},
})

PRESETS = {"day": DAY, "night": NIGHT, "twilight": TWILIGHT}

# Studio options -> preset overrides
DAY_SKY = {"original": None, "clouds": "clouds", "clear": "clear"}
TWILIGHT_SKY = {"original": None, "clear": "twilight_clear", "clouds": "twilight_clouds"}


def job_overrides(style: str, sky: str = "original", look: str = "natural", lights: bool = True) -> dict:
    """Overrides for one shoot, from the options chosen in the studio.

    day:      sky = original | clouds | clear
    twilight: look = natural | purple, sky = original | clear | clouds,
              lights = enhance every interior/exterior light"""
    ov: dict = {}
    if style == "day":
        ov["sky_dome"] = DAY_SKY.get(sky)
    else:
        dome = TWILIGHT_SKY.get(sky)
        ov["sky_dome"] = dome
        if dome:
            # the dome gives the sky its shape; the house twilight colour is painted
            # over it after the grade (skydome.GRADIENTS)
            ov["sky_gradient"] = "purple" if look == "purple" else "natural"
        elif look == "purple":
            ov["sky_purple_deg"] = 18.0
            ov["sky_purple_sat"] = 0.08
        ov["lights"] = bool(lights)
    return ov
