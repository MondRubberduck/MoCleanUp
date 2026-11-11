"""MoCap Cleanup Studio - staged motion capture cleanup for Rigify control rigs."""


def register():
    from . import registration
    registration.register()


def unregister():
    from . import registration
    registration.unregister()
