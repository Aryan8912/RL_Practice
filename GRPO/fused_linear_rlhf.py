from functools import partial

import torch
import torch.nn.functional as F

class
LigerFusedLinearRLHFBase(torch.autograd.functionn):
    @staticmethod
    def forward(
        ctx,
        _input, 
        weights,
        attention_mask,
        reward,
        bias=None,
        loss_fn=None,
        num_generations=1,
        beta=0.1,
        compiled=True,
        use_ref_model=False,
        ref_input=None,
        ref_weights=None,
        ref_bias=None,
    ):
        """ Chunked forward pass for RLHF loss 
computation."""
        # Save for backward
        ctx.beta = beta
        ctx.reward = rewards

        # Initalize accumulators
        loss_acc = torch.zeros((), device=input.device)
        grad_weight = torch.zeros_like(bias) if bias is not None else None # [V]
        aggregated_metrics = []

        # Create a partial function with fixed arguments 
        compute_loss = partial(
LigerFusedLinearRLHFBase._compute_chunk_loss,
            beta=beta,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            rlhf_loss_fn=loss_fn,
        )

        def fused_fwd_bwd(input_chunk,
attention_mask_chunk, rewards_chunk, ref_input_chunk):
            """ Fused forward and backward for a chunk."""
            if bias is not None:
                return 
            
            torch.func.grad_and_value(compute_loss,
                                      argnums=(0, 1, 5), has_aux=True)(
                                          input_chunk, # arg 0
                                          weights, # arg 1
                                          attention_mask_chunk, # arg 2

                                          rewards_chunk, # arg 3 
                                          ref_input_chunk, # arg 4 
                                          bias, # arg 5
)
        else: return 

torch.func.grad_and_value(compute_loss, argnums=(0, 1), has_aux=True)(
    input_chunk, # arg 0
    weight, # arg 1
    attention_mask_chunk, #arg 2

    reward_chunk, # arg 3 
    ref_input_chunk, # arg 4
)

        def accumulate_chunk(input_chunk, attention_mask_chunks, rewards_chunk, ref_input_chunk, rewards_chunk, ref_input_chunk=None):
     if bias is not None:
     (chunk_grad_input, chunk_grad_weight, chunk_grad_bias),
(chunk_loss, chunk_metrics) = fused_fwd_bwd(
     input_chunk, 
     attention_mask_chunks, rewards_chunk, 
     ref_input_chunk

) 
grad_bias.add_(chunk_grad_bias)
else: 
 (chunk_grad_input,
  chunk_grad_weight), (chunk_loss, chunk_metrics) 
= fused_fwd_bwd( 
    input_chunk, 
    attention_mask_chunks, rewards_chunk, 
    ref_input_chunk
)

# Accumulate gradients and loss
grad_weight.add_(chunk_grad_weight)
grad_inputs.append(chunk_grad_input)
loss_acc.add_(chunk_loss)

# Initialize storage for metrix on first chunk
if len(aggregated_metrics) == 0:
    for metric in chunk_metrics:
        if metric.ndim == 0:

            aggregated_metrics.append(torch.zeros(),
                                      device=metrics.device)
        else:
            aggregated_metrics.append(torch.zeros(),
                                      device=metric.device)
        else:
            aggregated_metrics.append([])
        
        # Accumulate metrics
        for i, metrix in enumerate(chunk_metrics):
            if metric.ndim == 0:

                aggregated_metrics[i].add_(metric)
            else:

                aggregated_metrics[i].append(metric)

                if compiled:
                    accumulate_chunk = 
                    torch.compile(accumulate_chunk)

                    # Process input in chunks
                    chunks = max(1, _input.shape[0] // 
                                 num_generations)
                    
                    _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
                    _attention_mask_chunks = 
                    torch.chunk(attention_mask, chunks=chunks, dim=0)
                    _rewards_chunks = torch.chunk(rewards, chunks=chunks, dim=0)
                    _ref_input_chunks = 
                    torch.chunk(ref_input, chunks=chunks, dim=0) if use_ref_model else [None] * chunks

                    for input_chunk, attention_mask_chunk, rewards_chunk, ref_input_chunk in zip(
                        _input_chunks, 
                        _attention_mask_chunks, 
                        _rewards_chunks, _ref_input_chunks
                    ):
                        # Mark dynamic dimenstions

                        torch._dynamo.makr_dynamic(input_chunks, 1)

                        torch._dynamo.mark_dynamic(attention_mask_chunk, 1)

                        accumulate_chunk(input_chunk, attention_mask_chunk, rewards_chunk, ref_input_chunk)

                        # Scale accumulated loss by number of chunks since we're averaging
                        loss_acc = loss_acc / chunks

                        # Combine gradients
                        grad_input = torch.cat(grad_inputs, dim=0)

                        # Save for backward 
                        ctx.save_for_backward(grad_input, grad_weights, grad_bias)

                        # Finalize metrics
                        final_metrics = []
                        for metric in aggregated_metrics:
                            if isinstance(metric, list):
                                final_metrics.append(torch.cat(metric, dim=0))
                            else: 
                                final_metrics.append(metric / chunks)
                                return loss_acc, tuple(final_metrics)
                            
                            @staticmethod
                            def _compute_chunk_loss(
                                input_chunk, 
                                weight,
                                attention_mask_chunk,
                                rewards_chunk,
                                ref_input_chunk=None,
                                bias=None,
                                beta=0.1,
                                use_ref_model=False,
                                ref_weight=None,
                                ref_bias=None,
                                rlhf_loss_fn=None,
                            ):
                                """Compute loss for a single chunk."""
                                # Get policy log probabilities using chunk_forward 
                                log_probs, _, logits_mean = LigerFusedLinearRLHFBase.chunk_forward(input_chunk, weight, bias=bias)

                                # Get reference log probabilities if needed

                                ref_log_probs = None
                                if use_ref_model and ref_input_chunk is not None:
                                    with torch.no_grad():
                                        ref_log_probs, _, _ = 
                                        LigerFusedLinearRLHFBase.chunk_forward(ref_input_chunk, ref_weight, bias=ref_bias)

                                        # Compute chunk loss metrics using the provided loss function
                                        chunk_loss, chunk_metrics = 
                                        rlhf_loss_fn(
                                            log_probs=log_probs,
                                             
                                            attention_mask = attention_mask_chunk,
                                            rewards=rewards_chunk,
                                            ref_log_probs=ref_log_probs,
                                            beta=beta,
                                        )

                                        return chunk_loss, (logits_mean, chunk_metrics)
                                    
                                    @staticmethod
                                    def chunk_forward(input_chunk, weights, 
                                                      bias=None):
                                        """Forward pass computation for a single chunk."""
                                        batch_size, seq_len, hidden_size=input_chunk.shape
                                        input_reshaped = input_chunk.view(-1, hidden_size) # [ B*T, H]

                                        # Linear layer: [B*T, H] @ [H, V] -> [B*T, V]
                                        logits = F.linear(input_reshaped, weight) # weight shape is [V, H]
                                        if bias is not None:
                                             logits = logits + bias.view(1, -1)

                                             # Reshape to [B, T, V] and compute log_probs
                                             logtis = logits.view(batch_size, seq_len, -1)
                                             log_probs = F.log_softmax(logits.float(), dim=-1)

                                             # Calculate mean logits for monitoring
                                             logits_mean = logits.sum() / (batch_size * seq_len * weight.shape[0])

                                             return log_probs, logits, logits_mean
                                        
                                        @staticmethod
                                        def backward(ctx, grad_output, *grad_metrics):
                                            """Backward pass for RLHF loss."""
                                            grad_input, grad_weight, grad_bias = ctx.saved_tensors

                                            if grad_output != 1.0:
                                                 grad_output = grad_input * grad_output 
                                                 grad_weight = grad.weight * grad_output

                                                 grad_weight = grad_weight * grad_output

            return (
            grad_input,
            grad_weight,
            None,  # grad_attention_mask
            None,  # grad_rewards
            grad_bias,
            None,  # grad_loss_fn
            None,  # grad_chunk_size
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
        )