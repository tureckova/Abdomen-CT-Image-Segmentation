import numpy as np
import matplotlib.pyplot as plt

att0 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task03_Liver/nnUNetTrainer__nnUNetPlans/fold_1/attention/liver_10_att0.npy")
att1 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task03_Liver/nnUNetTrainer__nnUNetPlans/fold_1/attention/liver_10_att1.npy")
org = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task03_Liver/nnUNetTrainer__nnUNetPlans/fold_1/attention/liver_10_org.npy")

indx = 160
print("Index: {}".format(indx))
print("Im shape: {}".format(org.shape))
print("ATM shape: {}".format(att1.shape))
att0_disp1 = att0[0, indx, :, :]
im_disp = org[0, indx, :, :]
att1_disp1 = att1[0, indx, :, :]

plt.subplot(121)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att1_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - top level")
plt.subplot(122)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att0_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - second level")
plt.show()

att0 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task03_Liver/nnUNetTrainer__nnUNetPlans/fold_1/attention/liver_103_att0.npy")
att1 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task03_Liver/nnUNetTrainer__nnUNetPlans/fold_1/attention/liver_103_att1.npy")
org = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task03_Liver/nnUNetTrainer__nnUNetPlans/fold_1/attention/liver_103_org.npy")

indx = 160
print("Index: {}".format(indx))
print("Im shape: {}".format(org.shape))
print("ATM shape: {}".format(att1.shape))
att0_disp1 = att0[0, indx, :, :]
im_disp = org[0, indx, :, :]
att1_disp1 = att1[0, indx, :, :]

plt.subplot(121)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att1_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - top level")
plt.subplot(122)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att0_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - second level")
plt.show()

att0 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/KiTS2019/nnUNetTrainer__nnUNetPlans/fold_4/attention/KiTS2019_001_att0.npy")
att1 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/KiTS2019/nnUNetTrainer__nnUNetPlans/fold_4/attention/KiTS2019_001_att1.npy")
org = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/KiTS2019/nnUNetTrainer__nnUNetPlans/fold_4/attention/KiTS2019_001_org.npy")

indx = 37
print("Index: {}".format(indx))
print("Im shape: {}".format(org.shape))
print("ATM shape: {}".format(att1.shape))
att0_disp1 = np.rot90(att0[0, :, :, indx])
im_disp = np.rot90(org[0, :, :, indx])
att1_disp1 = np.rot90(att1[0, :, :, indx])

plt.subplot(121)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att1_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - top level")
plt.subplot(122)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att0_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - second level")
plt.show()

att0 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/KiTS2019/nnUNetTrainer__nnUNetPlans/fold_4/attention/KiTS2019_007_att0.npy")
att1 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/KiTS2019/nnUNetTrainer__nnUNetPlans/fold_4/attention/KiTS2019_007_att1.npy")
org = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/KiTS2019/nnUNetTrainer__nnUNetPlans/fold_4/attention/KiTS2019_007_org.npy")

indx = 37
print("Index: {}".format(indx))
print("Im shape: {}".format(org.shape))
print("ATM shape: {}".format(att1.shape))
att0_disp1 = np.rot90(att0[0, :, :, indx])
im_disp = np.rot90(org[0, :, :, indx])
att1_disp1 = np.rot90(att1[0, :, :, indx])

plt.subplot(121)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att1_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - top level")
plt.subplot(122)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att0_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - second level")
plt.show()

att0 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task07_Pancreas/nnUNetTrainer__nnUNetPlans/fold_3/attention/pancreas_001_att0.npy")
att1 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task07_Pancreas/nnUNetTrainer__nnUNetPlans/fold_3/attention/pancreas_001_att1.npy")
org = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task07_Pancreas/nnUNetTrainer__nnUNetPlans/fold_3/attention/pancreas_001_org.npy")

indx = 50
print("Index: {}".format(indx))
print("Im shape: {}".format(org.shape))
print("ATM shape: {}".format(att1.shape))
att0_disp1 = att0[0, indx, :, :]
im_disp = org[0, indx, :, :]
att1_disp1 = att1[0, indx, :, :]

plt.subplot(121)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att1_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - top level")
plt.subplot(122)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att0_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - second level")
plt.show()

att0 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task07_Pancreas/nnUNetTrainer__nnUNetPlans/fold_3/attention/pancreas_006_att0.npy")
att1 = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task07_Pancreas/nnUNetTrainer__nnUNetPlans/fold_3/attention/pancreas_006_att1.npy")
org = np.load("/home/tureckova/Pictures/nnUNet/nnUNet_output/nnUNet/3d_lowres/Task07_Pancreas/nnUNetTrainer__nnUNetPlans/fold_3/attention/pancreas_006_org.npy")

indx = 46
print("Index: {}".format(indx))
print("Im shape: {}".format(org.shape))
print("ATM shape: {}".format(att1.shape))
att0_disp1 = att0[0, indx, :, :]
im_disp = org[0, indx, :, :]
att1_disp1 = att1[0, indx, :, :]

plt.subplot(121)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att1_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - top level")
plt.subplot(122)
plt.imshow(im_disp, cmap='gray')
plt.imshow(att0_disp1, cmap='jet', alpha=0.5)
plt.title("Attention map - second level")
plt.show()