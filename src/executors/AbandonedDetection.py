
import os
import cv2
import sys
import numpy as np
import uuid
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.base.model import Detection, BoundingBox
from capsules.AbandonedDetection.src.utils.response import build_response
from capsules.AbandonedDetection.src.models.PackageModel import PackageModel
from capsules.AbandonedDetection.src.models.PackageModel import Image as ImageModel



class AbandonedDetection(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.alpha = self.request.get_param("alpha")
        self.beta = self.request.get_param("beta")
        self.ssim = self.request.get_param("ssim")
        self.tau = self.request.get_param("ssimThreshold")
        self.Q = self.request.get_param("ssimKernelSize")
        self.frame = self.request.get_param("inputImage")
        self.warning = self.request.get_param("warningRatio")
        self.inputMaskShort = self.request.get_param("inputMaskShort")
        self.inputMaskLong = self.request.get_param("inputMaskLong")
        self.max_energy = self.bootstrap.get("max_energy")
        self.warning_threshold = self.max_energy * self.warning
        uID = str(uuid.uuid4())
        self.image = ImageModel(name="Image_" + uID, uID=uID, mimeType="image/jpg", encoding="bytes", value=None, r_key='',
                           type="Image", timestamp=datetime.now().timestamp())


    @staticmethod
    def bootstrap(config: dict) -> dict:
        max_energy = 100
        return {"max_energy": max_energy, "heatmap":None}


    def compute_ssim_map(self,img1, img2):
        I1 = img1.astype(np.float32)
        I2 = img2.astype(np.float32)

        C1 = 6.5025
        C2 = 58.5225

        mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-6)

        return np.clip(ssim, 0, 1)

    def run(self):
        frame = Image.get_frame(self.frame, self.redis_db).value
        mask_short = Image.get_frame(self.inputMaskShort, self.redis_db).value
        mask_long = Image.get_frame(self.inputMaskLong, self.redis_db).value

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        current_frame = cv2.GaussianBlur(gray, (5, 5), 0)

        if mask_short.ndim == 3:
            mask_short = cv2.cvtColor(mask_short, cv2.COLOR_BGR2GRAY)
        if mask_long.ndim == 3:
            mask_long = cv2.cvtColor(mask_long, cv2.COLOR_BGR2GRAY)

        mask_short = mask_short.astype(np.uint8)
        mask_long = mask_long.astype(np.uint8)
        current_frame = current_frame.astype(np.uint8)

        if bool(self.ssim):
            ssim_map = self.compute_ssim_map(current_frame,mask_long)
            rsimm = cv2.blur(ssim_map, (self.Q, self.Q))
            rsimm = np.clip(rsimm, 0, 1)
            ssim_mask = (rsimm < self.tau)
            candidate = (
                    (mask_long == 255) &
                    (mask_short == 0) &
                    (ssim_mask)
            )
        else:
            candidate = (
                    (mask_long == 255) &
                    (mask_short == 0)
            )

        if self.bootstrap["heatmap"] is None:
            self.bootstrap["heatmap"]= np.zeros_like(current_frame, dtype=np.float32)

        self.bootstrap["heatmap"][candidate] += self.alpha
        self.bootstrap["heatmap"][~candidate] -= self.beta
        np.clip(self.bootstrap["heatmap"], 0, self.max_energy, out=self.bootstrap["heatmap"])
        thresh_map_u8 = self.bootstrap["heatmap"]

        _, thresh_map = cv2.threshold(
            thresh_map_u8, self.warning_threshold, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            thresh_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            self.detections.append(
                Detection(
                    boundingBox=BoundingBox(
                        left=x, top=y, width=w, height=h
                    ),
                    confidence=1.0,
                    classId=1,
                    classLabel="abandoned",
                    imgUID=self.uID,
                    keyPoints=[]
                )
            )

        self.image.value = thresh_map_u8
        self.image = Image.set_frame(
            img=self.image, package_uID=self.uID, redis_db=self.redis_db
        )
        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
