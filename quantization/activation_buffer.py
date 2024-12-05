import torch

def create_activation_buffers(obj, arg, buffer_dict):
    arg_str = arg.split("quantize_")[1]
    obj.register_buffer(arg_str, buffer_dict[arg_str])
    obj.register_buffer(f"{arg_str}_scale", torch.tensor(0.0))
    obj.register_buffer(f"{arg_str}_zero_point", torch.tensor([0]))

def create_attn_buffer_dict(batch_size, block_size, n_embd, n_head, n_kv_group):
    buffer_dict = {}
    buffer_dict["attn_act_input"] = torch.zeros(batch_size, block_size, n_embd)
    buffer_dict["attn_act_qk_mult_q_input"] = torch.zeros(batch_size, n_head, block_size, n_embd // n_head)
    buffer_dict["attn_act_qk_mult_k_input"] = torch.zeros(batch_size, n_kv_group, block_size, n_embd // n_head)
    buffer_dict["attn_act_softmax_input"] = torch.zeros(batch_size, n_head, block_size, block_size)
    buffer_dict["attn_act_pv_mult_p_input"] = torch.zeros(batch_size, n_head, block_size, block_size)
    buffer_dict["attn_act_pv_mult_v_input"] = torch.zeros(batch_size, n_kv_group, block_size, n_embd // n_head)
    buffer_dict["attn_act_pv_mult_output"] = torch.zeros(batch_size, n_head, block_size, n_embd // n_head)
    buffer_dict["attn_act_output"] = torch.zeros(batch_size, block_size, n_embd)
    return buffer_dict

def create_mlp_buffer_dict(batch_size, block_size, n_embd, expansion_factor):
    buffer_dict = {}
    buffer_dict["mlp_act_input"] = torch.zeros(batch_size, block_size, n_embd)
    buffer_dict["mlp_act_activation_input"] = torch.zeros(batch_size, block_size, expansion_factor * n_embd)
    buffer_dict["mlp_act_activation_output"] = torch.zeros(batch_size, block_size, expansion_factor * n_embd)
    buffer_dict["mlp_act_output"] = torch.zeros(batch_size, block_size, n_embd)
    return buffer_dict