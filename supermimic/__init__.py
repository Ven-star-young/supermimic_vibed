"""SuperMimic motion tracking registered with mjlab."""

# Initialize mjlab's plugin discovery before importing any of our submodules.
# Otherwise importing an asset/MDP first can trigger discovery halfway through
# that module and expose a partially initialized module to task registration.
import mjlab as _mjlab  # noqa: F401
