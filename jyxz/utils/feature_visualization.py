import cv2
# import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from skimage import img_as_ubyte, img_as_float
class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.conv.cuda()

    def forward(self, x):
        x = self.conv(x)
        return x
def draw(x2, titles):
    # 创建一个空的热力图数据
    heatmaps = []
    # 针对每个通道生成热力图数据
    for channel in range(x2[0].shape[0]):
        # 创建一个单通道的热力图数据
        heatmap = x2[0][channel, :, :]
        # 将热力图数据进行归一化处理，以便在可视化时表现更好
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.float()
        # 将二维热力图扩展为三维（在通道维度上扩展）
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        # 将单通道热力图添加到列表中
        heatmaps.append(heatmap)
    # 将热力图数据列表合并为一个四维张量
    heat_data = torch.cat(heatmaps, dim=1).cuda()
    # 创建一个可视化窗口
    vis_win = vis.images(
        heat_data.view(-1, 1, heat_data.shape[2], heat_data.shape[3]),
        opts=dict(
            title=titles,
            caption='Channel Visualization',
        )
    )


def draw_mask(features,save_dir = 'F:/Project/CRNet1/visual_mask',name = None, dir = None):
    # features = torch.sigmoid(features)
    # print(features.size())
    conv = Conv1x1(64, 1)
    features = conv(features)
    # features = F.interpolate(features, size=(320, 320), mode='bilinear', align_corners=False)
    # print(features.size())
    # name = name.replace(".png", ".jpg")
    # features = features.cpu().numpy()
    # print(features.shape)
    save_dir = save_dir + '/' + dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(save_dir)
    # print(name)
    if os.path.exists(save_dir):
        # print("图片已成功保存在当前工作目录下")
        # features = features.astype(np.uint8)  # 将数据类型转换为float32
        # image = Image.fromarray(features)
        # image.save(save_dir, str(name))
        # np.save(os.path.join(save_dir, str(name)), features)
        features = (torch.sigmoid(features[0, 0])).cpu()
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        features = features.numpy()
        cv2.imwrite(os.path.join(save_dir, str(name)), img_as_ubyte(features))

def draw_edge(features,save_dir = 'F:/Project/CRNet1/visual_mask',name = None, dir = None):
    # features = torch.sigmoid(features)
    # print(features.size())
    # conv = Conv1x1(64, 1)
    # features = conv(features)
    # features = F.interpolate(features, size=(320, 320), mode='bilinear', align_corners=False)
    # print(features.size())
    # name = name.replace(".png", ".jpg")
    # features = features.cpu().numpy()
    # print(features.shape)
    save_dir = save_dir + '/' + dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(save_dir)
    # print(name)
    if os.path.exists(save_dir):
        # print("图片已成功保存在当前工作目录下")
        # features = features.astype(np.uint8)  # 将数据类型转换为float32
        # image = Image.fromarray(features)
        # image.save(save_dir, str(name))
        # np.save(os.path.join(save_dir, str(name)), features)
        # features = (torch.sigmoid(features[0, 0])).cpu()
        features = (features[0, 0]).cpu()
        # features = torch.sigmoid(features)
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        # features = features / features.max()
        # features = features / features.max()
        features = features.numpy()
        cv2.imwrite(os.path.join(save_dir, str(name)), img_as_ubyte(features))

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap)-np.min(heatmap) + 1e-8)
    # heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps


def featuremap_2_heatmap_2map(feature_map, feas):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    feas = feas.detach().cpu().numpy()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.maximum(np.max(heatmap), np.max(feas))
    heatmaps.append(heatmap)

    return heatmaps


def featuremap_2_heatmap_channel(feature_map, feature_map_all):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    feature_map_all = feature_map_all.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(feature_map_all)
    heatmaps.append(heatmap)
    return heatmaps


def featuremap_2_heatmap_channel_2(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    # feature_map_all = feature_map_all.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(feature_map)
    heatmaps.append(heatmap)
    return heatmaps


def draw_feature_map(features,save_dir = './heatmap',name = None, dir = 'real'):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (80, 80))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # cv2.imshow("1", superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # save_dir = save_dir + '/' + dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # print(save_dir)
                # print(name)
                # print(superimposed_img.shape)
                cv2.imwrite(os.path.join(save_dir, str(name)), superimposed_img)
                # j = j + 1
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,str(name)), superimposed_img)
                # i=i+1


def draw_feature_map_channel(features,save_dir = 'F:/Project/CRNet1/heatmap',name = None, dir = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            save_dir_image = os.path.join(save_dir, dir, name)
            if not os.path.exists(save_dir_image):
                os.makedirs(save_dir_image)
            for c in range(heat_maps.size(1)):
                heatmap = heat_maps[:, c]
                # print(str(dir)+'----'+str(c)+':'+str(heatmap.max())+'----'+str(heatmap.min()))
                heatmap = heatmap.unsqueeze(0)
                heatmaps = featuremap_2_heatmap_channel(heatmap, heat_maps)
                # heatmaps = featuremap_2_heatmap(heatmap)
                for h_map in heatmaps:
                    h_map = np.uint8(255 * h_map)
                    h_map = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
                    superimposed_img = h_map
                    cv2.imwrite(os.path.join(save_dir_image, f'{name}_channel_{c}.png'), superimposed_img)
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,str(name)), superimposed_img)
                # i=i+1


def draw_gary(features,save_dir = './GrayMap',name = None, dir = None):
    save_dir_image = os.path.join(save_dir, dir, name)
    os.makedirs(os.path.dirname(save_dir_image), exist_ok=True)  # 创建目录
    # 将特征张量转换为灰度图像
    gray_img = torch.squeeze(features[:, 0, :, :]).detach().cpu().numpy()
    # 将灰度图像转换为PIL图像对象
    pil_img = Image.fromarray(gray_img.astype(np.uint8), mode='L')
    pil_img.save(save_dir_image)


def draw_feature_map_channel_2(features,features2, save_dir = 'F:/Project/CRNet1/heatmap',name = None, dir = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps,heat_maps2 in zip(features, features2):
            heat_maps = heat_maps.unsqueeze(0)
            heat_maps2 = heat_maps2.unsqueeze(0)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            save_dir_image = os.path.join(save_dir, dir, name)
            if not os.path.exists(save_dir_image):
                os.makedirs(save_dir_image)
            for c in range(heat_maps.size(1)):
                heatmap = heat_maps[:, c]
                heatmap2 = heat_maps2[:, c]
                # print(str(dir)+'----'+str(c)+':'+str(heatmap.max())+'----'+str(heatmap.min()))
                heatmap = heatmap.unsqueeze(0)
                heatmap2 = heatmap2.unsqueeze(0)
                heatmaps = featuremap_2_heatmap_channel(heatmap, heatmap2)
                for h_map in heatmaps:
                    h_map = np.uint8(255 * h_map)
                    h_map = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
                    superimposed_img = h_map
                    cv2.imwrite(os.path.join(save_dir_image, f'{name}_channel_{c}.png'), superimposed_img)
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,str(name)), superimposed_img)
                # i=i+1


def draw_CALayer(feature, save_dir = 'F:/Project/CRNet1/heatmap',name = None, dir = None):
    # 假设加权后的特征图的形状为 (1, C, H, W)
    # weighted_feature_map 是加权后的特征图的数据
    # 调整特征图的形状，去掉批次维度，使其适配 matplotlib 的展示
    max = feature.max().cpu().numpy()
    min = feature.min().cpu().numpy()
    feature = feature.squeeze()
    feature = feature.cpu().numpy()
    # 设置保存路径
    save_path = os.path.join(save_dir, dir, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 循环遍历通道数并绘制每个通道的灰度图
    # plt.figure(figsize=(10, 5))
    for channel in range(feature.shape[0]):
        # plt.subplot(1, feature.shape[0], channel + 1)
        # plt.imshow(feature[channel, :, :], cmap='gray')
        # plt.axis('off')

        # 构造保存文件名
        filename = f"channel_{channel}.png"
        # 拼接完整的保存路径
        save_file = os.path.join(save_path, filename)
        # print(feature[channel, :, :])
        # 保存灰度图像
        # cv2.imwrite(save_file, ((feature[channel, :, :]-min)/(max))*255)
        cv2.imwrite(save_file, (feature[channel, :, :]) * 255)

    # 显示图像窗口
    plt.show()


def drawCAM(image, features, save_dir='F:/Project/CRNet1/heatmap', name=None, dir=None):
    i = 0
    if isinstance(features, torch.Tensor):
        for heat_maps in features:
            heat_maps = heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (80, 80))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # cv2.imshow("1", superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                save_dir = save_dir + '/' + dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # print(save_dir)
                # print(name)
                # print(superimposed_img.shape)
                # print(image.shape)
                # print(superimposed_img.shape)
                superimposed_img = cv2.addWeighted(image, 0.6, superimposed_img, 0.4, 0)
                cv2.imwrite(os.path.join(save_dir, str(name)), superimposed_img)
                # j = j + 1
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir, str(name)), superimposed_img)
                # i=i+1


def draw_feature_map_control_max(features,save_dir = 'F:/Project/CRNet1/heatmap',name = None, dir = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (80, 80))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # cv2.imshow("1", superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                save_dir = save_dir + '/' + dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # print(save_dir)
                # print(name)
                # print(superimposed_img.shape)
                cv2.imwrite(os.path.join(save_dir, str(name)), superimposed_img)
                # j = j + 1
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,str(name)), superimposed_img)
                # i=i+1


def draw_feature_map_2map(feas, features,save_dir = 'F:/Project/CRNet1/heatmap',name = None, dir = None):
    i=0

    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap_2map(heat_maps, feas)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (80, 80))
            for heatmap in heatmaps:
                print(1)
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # cv2.imshow("1", superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                save_dir = save_dir + '/' + dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # print(save_dir)
                # print(name)
                # print(superimposed_img.shape)
                cv2.imwrite(os.path.join(save_dir, str(name)), superimposed_img)
                # j = j + 1
        # for heat_mps in feas:
        #     heat_mps=heat_mps.unsqueeze(0)
        #     heatmps = featuremap_2_heatmap_2map(feas, heat_mps)
        #     # 这里的h,w指的是你想要把特征图resize成多大的尺寸
        #     # heatmap = cv2.resize(heatmap, (80, 80))
        #     for heatmp in heatmps:
        #         heatmp = np.uint8(255 * heatmp)
        #         # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
        #         heatmp = cv2.applyColorMap(heatmp, cv2.COLORMAP_JET)
        #         superimposed_img = heatmp
        #         # plt.imshow(superimposed_img,cmap='gray')
        #         # plt.show()
        #         # cv2.imshow("1", superimposed_img)
        #         # cv2.waitKey(0)
        #         # cv2.destroyAllWindows()
        #         save_dir = save_dir + '/' + dir
        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)
        #         # print(save_dir)
        #         # print(name)
        #         # print(superimposed_img.shape)
        #         cv2.imwrite(os.path.join(save_dir, str(name)), superimposed_img)
        #         # j = j + 1
    else:
        for featuremap in features:
            print(2)
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,str(name)), superimposed_img)
                # i=i+1