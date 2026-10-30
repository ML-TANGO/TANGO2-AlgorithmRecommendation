"""Supervised next-token loss without materializing a sequence x vocabulary tensor."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def answer_loss(model,inputs,chunk_size=128,num_items_in_batch=None):
    # Single-device Trainer only. PEFT injects LoRA layers into this same backbone.
    core=model.get_base_model() if hasattr(model,'peft_config') else model
    labels=inputs['labels']
    outputs=core.model(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask'),use_cache=False,return_dict=True)
    hidden=outputs.last_hidden_state[:,:-1,:].reshape(-1,outputs.last_hidden_state.shape[-1])
    targets=labels[:,1:].reshape(-1)
    selected=targets.ne(-100).nonzero(as_tuple=True)[0]
    if not selected.numel():raise ValueError('No supervised next-token targets')
    total=hidden.new_zeros((),dtype=torch.float32)
    softcap=getattr(core.config,'final_logit_softcapping',None)
    def loss_chunk(h,y):
        logits=core.lm_head(h)
        if softcap is not None:logits=torch.tanh(logits/softcap)*softcap
        return F.cross_entropy(logits.float(),y,reduction='sum')
    for positions in selected.split(chunk_size):
        h=hidden[positions];y=targets[positions]
        part=checkpoint(loss_chunk,h,y,use_reentrant=False) if model.training and torch.is_grad_enabled() else loss_chunk(h,y)
        total=total+part
    denominator=num_items_in_batch if num_items_in_batch is not None else selected.numel()
    return total/torch.as_tensor(denominator,device=total.device)
