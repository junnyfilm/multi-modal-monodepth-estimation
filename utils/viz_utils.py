import numpy as np
import os
import cv2

def to_np(data):
    try:
        return data.cpu().numpy()
    except:
        return data.detach().cpu().numpy()


class Visualize_CV(object):
    def __init__(self):
        self.height = 480
        self.width = 640

        self.line = np.zeros((self.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

    def update_image(self, img, name='img'):
        self.show[name] = img

    def saveimg(self, dir_name, file_name, show_list):
        # boundary line
        if self.show[show_list[0]].shape[0] != self.line.shape[0]:
            self.line = np.zeros((self.show[show_list[0]].shape[0], 3, 3), dtype=np.uint8)
            self.line[:, :, :] = 255
        disp = self.line

        for i in range(len(show_list)):
            if show_list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[show_list[i]], self.line), axis=1)

        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def savetxt(self, dir_name, file_name, matrices):
        txt_name = file_name.replace('png', 'txt')
        os.makedirs(dir_name, exist_ok=True)
        f = open(os.path.join(dir_name, txt_name), 'w')
        for i, line in enumerate(matrices):
            if i == 0:
                f.write('GT\n')
            if i == 3:
                f.write('\nPred\n')
            f.write(f'{line[0]:.7f}  {line[1]:.7f}  {line[2]:.7f}  {line[3]:.7f}\n')
        f.close()

    def save_results(self, frame1, frame2, gt1, gt2, pred1, pred2, error1, error2, save_dir, img_name):
        ######################################################################################
        # Visualize
        self.update_image(img=frame1, name='frame1')
        self.update_image(img=frame2, name='frame2')
        self.update_image(img=gt1, name='gt1')
        self.update_image(img=gt2, name='gt2')
        self.update_image(img=pred1, name='pred1')
        self.update_image(img=pred2, name='pred2')
        self.update_image(img=error1, name='error1')
        self.update_image(img=error2, name='error2')

        ######################################################################################
        # Save visualize results
        self.saveimg(dir_name=save_dir, file_name=img_name, show_list=['frame1', 'frame2', 'gt1', 'gt2', 'pred1', 'pred2', 'error1', 'error2'])

    def save_pos_results(self, R_gt, R_pred, T_gt, T_pred, save_dir, img_name):
        R_gt = to_np(R_gt)
        R_pred = to_np(R_pred)
        R_gt = R_gt.reshape(3, 3)          # Rotation matrix
        R_pred = R_pred.reshape(3, 3)      # Rotation matrix

        T_gt = to_np(T_gt)
        T_pred = to_np(T_pred)
        T_gt = T_gt.reshape(3, 1)          # Rotation matrix
        T_pred = T_pred.reshape(3, 1)      # Rotation matrix

        gt_matrix = np.concatenate([R_gt, T_gt], axis=1)          # [3, 4]
        pred_matrix = np.concatenate([R_pred, T_pred], axis=1)    # [3, 4]

        matrices = np.concatenate([gt_matrix, pred_matrix], axis=0)         # [3+3, 4]

        self.savetxt(dir_name=save_dir, file_name=img_name, matrices=matrices)
