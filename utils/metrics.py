# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import torch


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10': log10.item(), 'silog': silog.item()}


def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                            352, left_margin:left_margin + 1216]            

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif args.dataset == 'nyudepthv2':
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[45:471, 41:601] = 1
    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]

def eval_pose(pred, target):
    def make_batch_identity(batch_size, identity_dim):
        I_matrix = torch.eye(identity_dim)
        I_matrix = I_matrix.reshape((1, identity_dim, identity_dim))
        batch_identity = I_matrix.repeat(batch_size, 1, 1)
        return batch_identity
    
    B = pred['R12'].size(dim=0)

    if pred['R12'] is None:
        diff_r12 = pred['R12'].view(B,-1) - target['R12'].view(B,-1)
        diff_t12 = pred['T12'].view(B,-1) - target['T12'].view(B,-1)
        mse_r12 = torch.mean(torch.pow(diff_r12, 2))
        mse_t12 = torch.mean(torch.pow(diff_t12, 2))
        return {'pose_mse_r12': mse_r12.item(), 
                'pose_mse_t12': mse_t12.item(),
                'pose_mse_r21': 0.,
                'pose_mse_t21': 0.,
                'pose_mse_r_identity': 0.,
                'pose_mse_t_identity': 0.,
                }
    else:
        diff_r12 = pred['R12'].view(B,-1) - target['R12'].view(B,-1)
        diff_t12 = pred['T12'].view(B,-1) - target['T12'].view(B,-1)
        diff_r21 = pred['R21'].view(B,-1) - target['R21'].view(B,-1)
        diff_t21 = pred['T21'].view(B,-1) - target['T21'].view(B,-1)
        mse_r12 = torch.mean(torch.pow(diff_r12, 2))
        mse_t12 = torch.mean(torch.pow(diff_t12, 2))
        mse_r21 = torch.mean(torch.pow(diff_r21, 2))
        mse_t21 = torch.mean(torch.pow(diff_t21, 2))

        batch_identity = make_batch_identity(B, 3)
        pred_batch_identity = torch.matmul(pred['R12'].view(B,3,3), pred['R21'].view(B,3,3)).detach().cpu()
        diff_r_identity = pred_batch_identity - batch_identity
        mse_r_identity = torch.mean(torch.pow(diff_r_identity, 2))
        diff_t_identity = pred['T12'].view(B,3,1) + torch.matmul(pred['R12'].view(B,3,3), pred['T21'].view(B,3,1))
        mse_t_identity = torch.mean(torch.pow(diff_t_identity, 2))

        return {'pose_mse_r12': mse_r12.item(), 
                'pose_mse_t12': mse_t12.item(),
                'pose_mse_r21': mse_r21.item(),
                'pose_mse_t21': mse_t21.item(),
                'pose_mse_r_identity': mse_r_identity.item(),
                'pose_mse_t_identity': mse_t_identity.item(),
                }
