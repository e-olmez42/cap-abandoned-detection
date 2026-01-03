
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
        self.inputImageOne = self.request.get_param("inputImageOne")
        self.inputImageTwo = self.request.get_param("inputImageTwo")
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

        img1 = Image.get_frame(self.inputImageOne, self.redis_db).value
        img2 = Image.get_frame(self.inputImageTwo, self.redis_db).value

        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        ssim_map = self.compute_ssim_map(img1, img2)
        ssim_mask = (ssim_map < self.tau)

        if self.bootstrap["heatmap"] is None:
            self.bootstrap["heatmap"]= np.zeros_like(img1, dtype=np.float32)

        candidate = (img1 > 0) & ssim_mask

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
