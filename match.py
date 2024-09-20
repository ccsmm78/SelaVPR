import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import parser
import os
import network
import warnings
import time
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GeoLocalizationMatcher:
    def __init__(self, match_pattern="dense"):
        self.match_pattern = match_pattern
        self.args = parser.parse_arguments()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = self.load_model()

    def load_model(self):
        model = network.GeoLocalizationNet(self.args)
        model = model.to(self.args.device)
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(self.args.resume)["model_state_dict"]
        model.load_state_dict(state_dict)
        return model

    def get_patchfeature(self, imgpath):
        img = Image.open(imgpath)
        img = self.transform(img).unsqueeze(0).to(self.args.device)
        if self.match_pattern == "dense":
            feature, _ = self.model(img)
            feature = feature.view(1, 61 * 61, 128)
        elif self.match_pattern == "coarse":
            feature = self.model.module.backbone(img)
            feature = feature["x_norm_patchtokens"]
            feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
        return feature

    def get_keypoints(self, img_size):
        H, W = img_size
        if self.match_pattern == "dense":
            patch_size = 224 / 61
        elif self.match_pattern == "coarse":
            patch_size = 14
        N_h = int(H / patch_size)
        N_w = int(W / patch_size)
        keypoints = np.zeros((2, N_h * N_w), dtype=int)  # (x, y)
        keypoints[0] = np.tile(np.linspace(patch_size // 2, W - patch_size // 2, N_w, dtype=int), N_h)
        keypoints[1] = np.repeat(np.linspace(patch_size // 2, H - patch_size // 2, N_h, dtype=int), N_w)
        return np.transpose(keypoints)

    def match_batch_tensor(self, fm1, fm2, img_size, imgpath0, imgpath1):
        M = torch.matmul(fm2, fm1.T)  # (N, l, l)
        max1 = torch.argmax(M, dim=1)  # (N, l)
        max2 = torch.argmax(M, dim=2)  # (N, l)
        m = max2[torch.arange(M.shape[0]).reshape((-1, 1)), max1]  # (N, l)
        valid = torch.arange(M.shape[-1]).repeat((M.shape[0], 1)).cuda() == m  # (N, l) bool

        kps = self.get_keypoints(img_size)
        for i in range(fm2.shape[0]):
            idx1 = torch.nonzero(valid[i, :]).squeeze()
            idx2 = max1[i, :][idx1]
            assert idx1.shape == idx2.shape

            thetaGT, mask = cv2.findFundamentalMat(kps[idx1.cpu().numpy()], kps[idx2.cpu().numpy()], cv2.FM_RANSAC,
                                                   ransacReprojThreshold=5)
            idx1 = idx1[np.where(mask == 1)[0]]
            idx2 = idx2[np.where(mask == 1)[0]]

            cv_im_one = cv2.resize(cv2.imread(imgpath0), (224, 224))
            cv_im_two = cv2.resize(cv2.imread(imgpath1), (224, 224))

            inlier_keypoints_one = kps[idx1.cpu().numpy()]
            inlier_keypoints_two = kps[idx2.cpu().numpy()]
            kp_all1 = []
            kp_all2 = []
            matches_all = []
            print("Number of matched point pairs:", len(inlier_keypoints_one))

            for k in range(inlier_keypoints_one.shape[0]):
                kp_all1.append(cv2.KeyPoint(inlier_keypoints_one[k, 0].astype(float), inlier_keypoints_one[k, 1].astype(float), 1, -1, 0, 0, -1))
                kp_all2.append(cv2.KeyPoint(inlier_keypoints_two[k, 0].astype(float), inlier_keypoints_two[k, 1].astype(float), 1, -1, 0, 0, -1))
                matches_all.append(cv2.DMatch(k, k, 0))

            im_allpatch_matches = cv2.drawMatches(cv_im_one, kp_all1, cv_im_two, kp_all2,
                                                  matches_all, None, matchColor=(0, 255, 0), flags=2)
            cv2.imwrite("patch_matches.jpg", im_allpatch_matches)

            # Display the image using imshow
            cv2.imshow('Patch Matches', im_allpatch_matches)
            cv2.waitKey(1)  # Wait for a key press to close the image window
            #cv2.destroyAllWindows()

def do_match(imgpath0, imgpath1, matcher):
    start_time = time.time()
    # Patch feature 추출
    patch_time = time.time()
    patch_feature0 = matcher.get_patchfeature(imgpath0)
    patch_feature1 = matcher.get_patchfeature(imgpath1)
    print(f"Patch feature extraction time: {time.time() - patch_time:.2f} seconds")

    # 패치 토큰 크기 출력
    print("Size of patch tokens:", patch_feature1.shape[1:])

    # 매칭 수행
    match_time = time.time()
    matcher.match_batch_tensor(patch_feature0[0], patch_feature1, img_size=(224, 224), imgpath0=imgpath0, imgpath1=imgpath1)
    print(f"Matching time: {time.time() - match_time:.2f} seconds")

    # 전체 실행 시간 출력
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


def main():
    start_time = time.time()
    
    # Matcher 클래스 생성
    matcher = GeoLocalizationMatcher(match_pattern="coarse")  #"dense", "coarse"
    print(f"Model loading time: {time.time() - start_time:.2f} seconds")

    for i in range(1, 3):
        # 이미지 경로 설정
        imgpath0 = f"./image/img_pair/img0.jpg"
        imgpath1 = f"./image/img_pair/img{i}.jpg"
        do_match(imgpath0, imgpath1, matcher)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

