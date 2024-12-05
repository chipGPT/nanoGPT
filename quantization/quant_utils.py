def set_variant(variant, default_variant):
    # If variant is false or None, then set to provided default value
    if not variant:
        return default_variant
    return variant

def create_activation_buffers(obj, arg):
    arg_str = arg.split("quantize_")[1]
    obj.register_buffer(arg_str, None)
    obj.register_buffer(f"{arg_str}_scale", None)
    obj.register_buffer(f"{arg_str}_zero_point", None)
