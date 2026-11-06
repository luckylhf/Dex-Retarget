"""Final geometric acceptance only; solver accuracy and motor bounds stay unchanged."""
import math

SUPPORT_ACCEPTANCE_M = 1e-3

def support_constraint_failed(minimum_z, contact_error_m):
    return (not math.isfinite(minimum_z) or not math.isfinite(contact_error_m)
            or minimum_z < -SUPPORT_ACCEPTANCE_M
            or abs(contact_error_m) > SUPPORT_ACCEPTANCE_M)
