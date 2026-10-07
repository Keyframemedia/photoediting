"""Signature looks. Values are tuned on real Keyframe brackets; see README."""

DAY = {
    "name": "day",
    # exposure: scene median (windows/sky excluded) to `key`, plus bias
    "exposure_mode": "median", "key": 0.27, "key_pct": 50, "key_exclude_top": 0.10,
    "exposure_bias": 0.0,
    # white balance: neutral surfaces land on a warm-clean white (Planckian target)
    "wb_target_cct": 5000, "wb_strength": 0.75, "wb_tint": 0.0, "wb_max_mired": 45,
    "wb_max_warm_mired": 12,
    # local tone mapping: hat-weighted fusion of virtual exposures
    "fusion_mode": "hat", "fusion_evs": [-6.0, -4.5, -3.0, -1.5, 0.0, 1.0],
    "fusion_hat": [0.0, 0.06, 0.88, 0.99], "fusion_ev_prior": 0.5,
    "fusion_sigma": 0.22, "global_mix": 0.0,
    # levels + tone
    "levels": True, "black_pct": 0.3, "white_pct": 99.5, "white_target": 0.96,
    "max_white_stretch": 1.12,
    "black": 0.0, "white": 1.0, "contrast": 0.30, "pivot": 0.42, "toe": 0.04, "shoulder": 0.06,
    "clarity": 0.35, "clarity_sigma_frac": 0.012, "micro_contrast": 0.20, "micro_sigma_frac": 0.002,
    "desat_highlights": 0.3,
    # colour (Oklab units)
    "vibrance": 0.35, "saturation": 0.08, "warmth_b": 0.016, "tint_a": 0.0,
    "sky_sat": 0.10, "sky_deepen": 0.03, "green_warm_deg": -6.0, "green_sat": 0.05,
    "orange_protect": 0.5,
    # sharpening
    "sharpen_radius": 0.9, "sharpen_amount": 0.55, "web_sharpen_radius": 0.6,
    "web_sharpen_amount": 0.55, "sharpen_threshold": 0.006,
}

# Twilight / night: warm interior glow against a deep blue sky. White balance is
# fixed (not auto) so tungsten/LED interiors stay golden and the dusk sky blue.
# NOTE: tuned blind - needs a real twilight set to finalise.
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
})

PRESETS = {"day": DAY, "night": NIGHT}
