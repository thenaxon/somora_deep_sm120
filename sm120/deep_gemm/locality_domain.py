"""deep_gemm.locality_domain (SM120 shim): no MLOPart / SM locality domains on GeForce/workstation parts."""


def get_num_locality_domains() -> int:
    return 1


def is_localization_available() -> bool:
    return False


def localize(t, *args, **kwargs):
    return t


def is_localized(t) -> bool:
    return False


def destroy_localizer() -> None:
    pass


def get_balanced_sm_locality_domains():
    return None
