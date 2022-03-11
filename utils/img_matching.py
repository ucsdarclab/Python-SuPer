from abc import ABC, abstractmethod
import cv2
from math import ceil
import os

from models import pwcnet
from utils.config import *
from utils.utils import *

class Matcher(ABC):
    
    # Match features (within the valid regions) between img1 and img2.
    @abstractmethod
    def match_features(self, img1, val1, img2, val2, ID):
        # Target outputs: coordinates (y*HEIGHT+x, Nx2) of matched 
        # points in the two images, named match1 & match2.
        pass

# OpenCV feature matching functions.
class cv2Matcher(Matcher):

    def __init__(self, feature_type='sift'):

        # Init the feature detector.
        if feature_type == 'orb': # Initiate ORB detector
            self.detector = cv2.ORB_create()
        elif feature_type == 'sift': # Initiate SIFT detector
            self.detector = cv2.SIFT_create()

        # Init the feature matcher.
        self.bf = cv2.BFMatcher()
    
    def match_features(self, img1, img2, ID):

        if torch.is_tensor(img1): img1 = torch_to_numpy(img1).astype('uint8')
        if torch.is_tensor(img2): img2 = torch_to_numpy(img2).astype('uint8')
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Find the keypoints(kp) and descriptors(des) with SIFT.
        kp1, des1 = self.detector.detectAndCompute(img1,None)
        kp2, des2 = self.detector.detectAndCompute(img2,None)
        # BFMatcher with default params.
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to find good matches.
        matches1 = []
        matches2 = []
        if vis_img_matching:
            # good_matches = []
            filtered_good_matches = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                # if vis_img_matching: good_matches.append([m])
                
                # Get the coordinates of the matched points.
                (x1, y1) = kp1[m.queryIdx].pt
                (x2, y2) = kp2[m.trainIdx].pt
                x1 = round(x1)
                y1 = round(y1)
                x2 = round(x2)
                y2 = round(y2)

                # TODO: Valid map-based filtering? Better filtering?
                # Discard matches in the invalid image regions and specular reflection areas.
                match_region1 = img1[max(y1-5,0):min(y1+5,HEIGHT), max(x1-5,0):min(x1+5,WIDTH)]
                match_region2 = img2[max(y2-5,0):min(y2+5,HEIGHT), max(x2-5,0):min(x2+5,WIDTH)]
                if (match_region1>250).any() or (match_region2>250).any():
                    continue

                if vis_img_matching: filtered_good_matches.append([m])
                matches1.append(torch.tensor([[x1,y1]], device=dev))
                matches2.append(torch.tensor([[x2,y2]], device=dev))

                # TODO: Remove overlap matches.

        if vis_img_matching:
            # out = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=2)
            # out = np.concatenate([out, \
            #     cv2.drawMatchesKnn(img1,kp1,img2,kp2,filtered_good_matches,None,flags=2)], \
            #     axis=0)
            out = cv2.drawMatchesKnn(img1,kp1,img2,kp2,filtered_good_matches,None,flags=2)
            cv2.imwrite(os.path.join(F_img_matching,str(ID)+'.jpg'), out)

        if len(matches1) > 0:
            matches1 = torch.cat(matches1, axis=0)
            matches2 = torch.cat(matches2, axis=0)
        return matches1, matches2

class DeepMatcher():

    def __init__(self, phase):

        if phase == "train": 
            freeze_optical_flow_net=False
            saved_model = get_PWCNet_model("pt")
        elif phase == "test":
            freeze_optical_flow_net=True
            saved_model = get_PWCNet_model("5200")

        self.init_flow_net(saved_model, freeze_optical_flow_net)

    def init_flow_net(self, saved_model, freeze_optical_flow_net):

        # Init flow net: PWC-Net
        self.flow_net = pwcnet.PWCNet().cuda()

        # Load the pretrained model of PWC-Net
        pretrained_dict = torch.load(saved_model)
        # Load only optical flow part
        model_dict = self.flow_net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k.partition('flow_net.')[2]: v for k, v in pretrained_dict.items() if "flow_net" in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.flow_net.load_state_dict(model_dict)

        # for name, param in self.flow_net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.requires_grad)

        if freeze_optical_flow_net:
            for param in self.flow_net.parameters():
                param.requires_grad = False

    def match_features(self, prev_img, img, projdata, points, valid, ID, point_match=True, coord_match=False):
        
        def save_flow(flow):
            flow = flow.detach().cpu().numpy()
            mag, ang = cv2.cartToPolar(flow[0,0,...], flow[0,1,...])
            # Use Hue, Saturation, Value colour model 
            hsv = np.zeros((HEIGHT,WIDTH,3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imwrite(os.path.join(output_folder, "debug","{}.jpg".format(ID)), bgr)

        v, u = torch.round(projdata[:,1]).long(),torch.round(projdata[:,2]).long()
        coords = v*WIDTH+u

        valid_proj = torch.zeros_like(valid)
        valid_proj[coords] = True

        idx_map = -torch.ones((PIXEL_NUM,), dtype=int, device=dev)
        idx_map[coords] = projdata[:,0].long()

        # rescale the image size to be multiples of 64
        divisor = 64.
        H_ = int(ceil(HEIGHT/divisor) * divisor)
        W_ = int(ceil(WIDTH/divisor) * divisor)
        prev_img = cv2.resize(prev_img[:,:,::-1], (W_, H_)) # RGB--->BGR
        img = cv2.resize(img[:,:,::-1], (W_, H_))

        ## Estimate flow.
        prev_img = torch.as_tensor(prev_img, device=dev)/255.0
        img = torch.as_tensor(img, device=dev)/255.0

        prev_img = prev_img.permute(2,0,1).unsqueeze(0)
        img = img.permute(2,0,1).unsqueeze(0)

        flow2, _, _, _, _, _ = self.flow_net.forward(prev_img, img)
        # flow2: 1x2x128x160, flow3: 1x2x64x80, flow4: 1x2x32x40, flow5:1x2x16x20, 
        # flow6: 1x2x8x10, features2: 1x565x128x160
        flow = torch.nn.functional.interpolate(input=20.0 * flow2, \
            size=(HEIGHT, WIDTH), mode='bilinear', align_corners=False)
        flow[:,0,...] *= WIDTH/ float(W_)
        flow[:,1,...] *= HEIGHT/ float(H_)
        
        ## Apply dense flow to warp the source points to target frame.
        xy_coords = torch.cat([U.unsqueeze(0), V.unsqueeze(0)], 0).unsqueeze(0).type(tfdtype_) # (1, 2, 480, 640)
        # xy_coords[0, :, v, u] = torch.stack([projdata[:,2],projdata[:,1]], dim=0).double()
        
        # Apply the flow to pixel coordinates.
        # flow = torch.nn.functional.grid_sample(
        #     flow, xy_coords.permute(0, 2, 3, 1), padding_mode='zeros', align_corners=False
        # )
        xy_coords_warped = xy_coords + flow
        xy_coords_warped_out = xy_coords_warped.permute(0, 2, 3, 1).view(-1,2).clone()

        # Normalize to be between -1, and 1.
        # Since we use "align_corners=False", the boundaries of corner pixels
        # are -1 and 1, not their centers.
        xy_coords_warped[:,0,:,:] = (xy_coords_warped[:,0,:,:]) / (WIDTH - 1)
        xy_coords_warped[:,1,:,:] = (xy_coords_warped[:,1,:,:]) / (HEIGHT - 1)
        xy_coords_warped = xy_coords_warped * 2 - 1

        # Permute the warped coordinates to fit the grid_sample format.
        xy_coords_warped = xy_coords_warped.permute(0, 2, 3, 1)

        if coord_match and not point_match:
            save_flow(flow)
            return idx_map[valid_proj], xy_coords_warped_out[valid_proj]

        # Sample target points at computed pixel locations.
        target_points = points.view(HEIGHT,WIDTH,3).permute(2,0,1).unsqueeze(0)
        target_matches = torch.nn.functional.grid_sample(
            target_points, xy_coords_warped, padding_mode='zeros', align_corners=False
        ).permute(0,2,3,1).view(-1,3)

        # Estimate the validity of the interpolate target points.
        # target_valid = valid.double().view(HEIGHT,WIDTH,1).permute(2,0,1).unsqueeze(0)
        target_valid = valid.type(tfdtype_).view(HEIGHT,WIDTH,1).permute(2,0,1).unsqueeze(0)
        target_valid = torch.nn.functional.grid_sample(
            target_valid, xy_coords_warped, padding_mode='zeros', align_corners=False
        ).permute(0,2,3,1).view(-1)
        valid = target_valid >= 0.999

        valid &= valid_proj

        # wrap_img = torch.nn.functional.grid_sample(
        #     img[:,:,16:496,:].type(tfdtype_), xy_coords_warped, padding_mode='zeros', align_corners=False
        # )
        # source = (255*prev_img).permute(0,2,3,1)[0].cpu().numpy()
        # target = (255*img).permute(0,2,3,1)[0].cpu().numpy()
        # wrap_img = (255*wrap_img).permute(0,2,3,1)[0].cpu().numpy()
        # wrap_img[~valid.view(HEIGHT,WIDTH).cpu().numpy()] = 0
        # # cv2.imwrite(os.path.join(output_folder, "debug","{}.jpg".format(ID)), \
        # #     np.concatenate([source[16:496], target[16:496], wrap_img], axis=1))
        # cv2.imwrite(os.path.join(output_folder, "debug","{}.jpg".format(ID)), \
        #     np.concatenate([source[16:496], wrap_img], axis=0))
        save_flow(flow)
        
        if point_match and coord_match:
            # return valid[valid_proj].nonzero().squeeze(1), target_matches[valid], \
            #     idx_map[valid_proj], xy_coords_warped.view(-1,2)[valid_proj]
            return valid[valid_proj].nonzero().squeeze(1), target_matches[valid], \
                idx_map[valid_proj], xy_coords_warped_out[valid_proj]
        elif point_match:
            return idx_map[valid], target_matches[valid]