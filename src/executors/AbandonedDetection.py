
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
        self.tau=12
        self.frame = self.request.get_param("inputImage")
        self.inputMaskShort = self.request.get_param("inputMaskShort")
        self.inputMaskLong = self.request.get_param("inputMaskLong")
        self.max_energy = self.bootstrap.get("max_energy")
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

        if mask_short.ndim == 3:
            mask_short = cv2.cvtColor(mask_short, cv2.COLOR_BGR2GRAY)
        if mask_long.ndim == 3:
            mask_long = cv2.cvtColor(mask_long, cv2.COLOR_BGR2GRAY)

        mask_short = mask_short.astype(np.uint8)
        mask_long = mask_long.astype(np.uint8)
        gray = gray.astype(np.uint8)

        mask_short = (mask_short > 0)
        mask_long = (mask_long > 0)

        if bool(self.ssim):
            ssim_map = self.compute_ssim_map(mask_short.astype(np.uint8) * 255,
                                             mask_long.astype(np.uint8) * 255)
            ssim_mask = (ssim_map < self.tau)
            candidate = mask_short & (~mask_long) & ssim_mask
        else:
            candidate = mask_short & (~mask_long)

        if self.bootstrap["heatmap"] is None:
            self.bootstrap["heatmap"]= np.zeros_like(gray, dtype=np.float32)

        self.bootstrap["heatmap"][candidate] += self.alpha
        self.bootstrap["heatmap"][~candidate] -= self.beta
        np.clip(self.bootstrap["heatmap"], 0, self.max_energy, out=self.bootstrap["heatmap"])

        alarm_mask =self.bootstrap["heatmap"] > (0.7 * self.max_energy)
        alarm_mask = alarm_mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            alarm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:
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

        self.image.value = alarm_mask
        self.image = Image.set_frame(
            img=self.image, package_uID=self.uID, redis_db=self.redis_db
        )
        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
