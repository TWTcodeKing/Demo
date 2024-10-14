from collections import OrderedDict
import torch
import numpy as np


def split_model(global_parameters,model_rate,user_idx,rounds,args):
    idx_i = [None for _ in range(len(user_idx))]
    idx = [OrderedDict() for _ in range(len(user_idx))]
    # roll_idx = {}
    if "resnet" in args.model:
        for k, v in global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            # if 'residual.1' in k or 'residual.4' in k or 'conv1' in k:
                            if 'conv1' in k or 'conv2' in k:
                                if idx_i[user_idx[m]] is None:
                                    idx_i[user_idx[m]] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[user_idx[m]]
                                scaler_rate = model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                # # over lap is None temp
                                # if self.cfg['overlap'] is None:
                                if args.strategy == "hetero_fl":
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                else:
                                    roll = rounds % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                    # else:
                                    #     overlap = self.cfg['overlap']
                                    #     roll_idx[k] += int(local_output_size * (1 - overlap)) + 1
                                    #     roll_idx[k] = roll_idx[k] % local_output_size
                                    #     roll = roll_idx[k]
                                    #     model_idx = self.model_idxs[k]
                                    model_idx = torch.roll(model_idx, roll, -1)
                                    output_idx_i_m = model_idx[:local_output_size]
                                idx_i[user_idx[m]] = output_idx_i_m
                            # elif 'shortcut.1' in k:
                            elif 'shortcut.0' in k:
                                # input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                input_idx_i_m = idx[user_idx[m]][k.replace('shortcut.0', 'conv1')][1]
                                # input_idx_i_m = idx[m][k.replace('shortcut.1', 'residual.1')][1]
                                output_idx_i_m = idx_i[user_idx[m]]
                            elif 'fc' in k:
                                input_idx_i_m = idx_i[user_idx[m]]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                print(f"unvalid {k}")
                                raise ValueError('Not valid k')
                            idx[user_idx[m]][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'fc' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[user_idx[m]][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                elif "running" in parameter_type:
                    input_idx_i_m = idx_i[user_idx[m]]
                    idx[user_idx[m]][k] = input_idx_i_m
                    # print(model_rate[user_idx[m]])
                    # print(len(input_idx_i_m))
                else:
                    pass
    elif "vit" in args.model:
        for k, v in global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            # if "patch_embed" in k and ("proj_conv" in k or "rpe_conv" in k):
                            if "patch_embed" in k:
                                if idx_i[user_idx[m]] is None:
                                    idx_i[user_idx[m]] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[user_idx[m]]
                                scaler_rate = model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                if args.strategy == "hetero_fl":
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                else:
                                    roll = rounds % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx = torch.roll(model_idx, roll, -1)
                                    output_idx_i_m = model_idx[:local_output_size]
                                idx_i[user_idx[m]] = output_idx_i_m
                            # if 'embedding' in k.split('.')[-2]:
                            #     output_idx_i_m = torch.arange(output_size, device=v.device)
                            #     scaler_rate = model_rate[user_idx[m]]
                            #     local_input_size = int(np.ceil(input_size * scaler_rate))
                            #     input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
                            #     idx_i[user_idx[m]] = input_idx_i_m
                            # elif 'decoder' in k and 'linear2' in k:
                            #     input_idx_i_m = idx_i[user_idx[m]]
                            #     output_idx_i_m = torch.arange(output_size, device=v.device)
                            # elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            #     input_idx_i_m = idx_i[user_idx[m]]
                            #     scaler_rate = model_rate[user_idx[m]]
                            #     local_output_size = int(np.ceil(output_size // args.nh
                            #                                     * scaler_rate))
                            #     output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
                            #         args.nh, -1))[:, :local_output_size].reshape(-1)
                            #     idx_i[user_idx[m]] = output_idx_i_m
                            # else:
                            #     input_idx_i_m = idx_i[user_idx[m]]
                            #     scaler_rate = model_rate[user_idx[m]]
                            #     local_output_size = int(np.ceil(output_size * scaler_rate))
                            #     output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                            #     idx_i[user_idx[m]] = output_idx_i_m
                            elif "q_conv" in k or "v_conv" in k or "k_conv" in k:
                                input_idx_i_m = idx_i[user_idx[m]]
                                scaler_rate = model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size // args.num_heads
                                                                * scaler_rate))
                                if args.strategy == "hetero_fl":
                                    output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
                                        args.num_heads, -1))[:, :local_output_size].reshape(-1)
                                else:
                                    roll = rounds % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx = torch.roll(model_idx, roll, -1)
                                    output_idx_i_m = (model_idx.reshape(
                                        args.num_heads, -1))[:, :local_output_size].reshape(-1)
                                idx_i[user_idx[m]] = output_idx_i_m
                                # if "q_conv" in k:
                                #     print(output_idx_i_m.shape)
                            elif "head" in k:
                                input_idx_i_m = idx_i[user_idx[m]]
                                output_idx_i_m = torch.arange(output_size,device=v.device)
                            else:
                                input_idx_i_m = idx_i[user_idx[m]]
                                # output_idx_i_m = torch.arange(output_size, device=v.device)
                                scaler_rate = model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                if args.strategy == "hetero_fl":
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                else:
                                    roll = rounds % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx = torch.roll(model_idx, roll, -1)
                                    output_idx_i_m = model_idx[:local_output_size]
                                idx_i[user_idx[m]] = output_idx_i_m
                            # elif "fc2" in k or "fc1" in k or "attn.proj_conv" in k:
                            #     input_idx_i_m = idx_i[user_idx[m]]
                            #     # output_idx_i_m = torch.arange(output_size, device=v.device)
                            #     scaler_rate = model_rate[user_idx[m]]
                            #     local_output_size = int(np.ceil(output_size * scaler_rate))
                            #     output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                            #     idx_i[user_idx[m]] = output_idx_i_m
                            # else:
                            #     input_idx_i_m = idx_i[user_idx[m]]
                            #     output_idx_i_m = torch.arange(output_size,device=v.device)
                            #     pass
                            # else:  # mlp.fc1    attn.proj_conv
                            #     # here has some problem must be
                            #     input_idx_i_m = idx_i[user_idx[m]]
                            #     scaler_rate = model_rate[user_idx[m]]
                            #     local_output_size = int(np.ceil(output_size * scaler_rate))
                            #     output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                            #     idx_i[user_idx[m]] = output_idx_i_m
                            idx[user_idx[m]][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        # if "fc2" in k:
                        #     input_idx_i_m = idx_i[user_idx[m]]
                        #     idx[user_idx[m]][k] = input_idx_i_m
                        if "q_conv" in k or "v_conv" in k or "k_conv" in k:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                            if "v_conv" not in k:
                                # idx_i[user_idx[m]] = idx[user_idx[m]][k.replace('bias', 'weight')][1]
                                idx_i[user_idx[m]] = idx[user_idx[m]][k.replace('bias', 'weight')]
                        elif "head" in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        # elif "head" in k:
                        #     idx[user_idx[m]][k] = torch.arange(input_size,device=v.device)
                        #     pass
                        else:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                elif "running" in parameter_type:
                    input_idx_i_m = idx_i[user_idx[m]]
                    idx[user_idx[m]][k] = input_idx_i_m
                else:
                    pass
    elif "vgg" in args.model:
        for k, v in global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'layer' in k:
                                if idx_i[user_idx[m]] is None:
                                    idx_i[user_idx[m]] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[user_idx[m]]
                                scaler_rate = model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                # # over lap is None temp
                                # if self.cfg['overlap'] is None:
                                if args.strategy == "hetero_fl":
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                else:
                                    roll = rounds % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                    # else:
                                    #     overlap = self.cfg['overlap']
                                    #     roll_idx[k] += int(local_output_size * (1 - overlap)) + 1
                                    #     roll_idx[k] = roll_idx[k] % local_output_size
                                    #     roll = roll_idx[k]
                                    #     model_idx = self.model_idxs[k]
                                    model_idx = torch.roll(model_idx, roll, -1)
                                    output_idx_i_m = model_idx[:local_output_size]
                                idx_i[user_idx[m]] = output_idx_i_m
                            elif 'fc.1' in k:
                                scaler_rate = model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
                                output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                idx_i[user_idx[m]] = output_idx_i_m
                            elif "fc" in k:
                                scaler_rate = model_rate[user_idx[m]]
                                input_idx_i_m = idx_i[user_idx[m]]
                                if output_size > 100:
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                else:
                                    output_idx_i_m = torch.arange(output_size,device=v.device)
                            # elif "fc" in k:
                            #     scaler_rate = model_rate[user_idx[m]]
                            #     print(f"key:{k},rate:{scaler_rate},input_size:{input_size},output_size:{output_size}")
                            #     input_idx_i_m = torch.arange(input_size, device=v.device)
                            #     output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                print(f"unvalid {k}")
                                raise ValueError('Not valid k')
                            idx[user_idx[m]][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'fc' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[user_idx[m]][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[user_idx[m]]
                            idx[user_idx[m]][k] = input_idx_i_m
                elif "running" in parameter_type:
                    input_idx_i_m = idx_i[user_idx[m]]
                    idx[user_idx[m]][k] = input_idx_i_m
                    # print(model_rate[user_idx[m]])
                    # print(len(input_idx_i_m))
                else:
                    pass
    # how about split method for transformer

    return idx
