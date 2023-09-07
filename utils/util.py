import os
import torch


def normalize_rot_vector(rot_vector):
    bs, dim = rot_vector.shape
    normed_rot_vector = torch.zeros_like(rot_vector)

    rot_matrix = rot_vector.reshape(bs, 3, 3)
    for b_idx in range(bs):
        U, _, V = torch.linalg.svd(rot_matrix[b_idx], full_matrices=False)
        S = torch.eye(3).cuda()
        normed_rot_martix = torch.linalg.multi_dot((U, S, V))

        normed_rot_vector[b_idx] = normed_rot_martix.reshape(-1)

    return normed_rot_vector


def save_model(args, net, optimizer, epoch):
    save_dir = os.path.join(args.log_dir, 'train/ckpt')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, 'epoch_%02d_model.ckpt' % epoch))

def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model_state_dict' not in ckpt_dict and 'optimizer_state_dict' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model_state_dict']
    weights = {}
    for key, value in state_dict.items():
        if key.find('.self_attn_weight') < 0:
            if key.startswith('module.'):
                weights[key[len('module.'):]] = value
            else:
                weights[key] = value
    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state)
    
    return ckpt_dict['epoch']

def save_model_best_rmse_model(args, net, optimizer, epoch, rmse, best_rmse):
    save_dir = os.path.join(args.log_dir, 'train/ckpt')
    os.makedirs(save_dir, exist_ok=True)

    if best_rmse > rmse:
        best_rmse = rmse

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, f'checkpoint_best.pth'))
        print('Saved best_model to ' + os.path.join(save_dir, f'checkpoint_best.pth'))
        print(f'Epoch [{epoch:03d}] Best model performances : ' + f'[rmse {rmse:.5f}]')
    return best_rmse
