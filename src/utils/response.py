
from sdks.novavision.src.helper.package import PackageHelper
from capsules.AbandonedDetection.src.models.PackageModel import PackageModel,OutputDetections, PackageConfigs, ConfigExecutor, AbandonedDetectionOutputs, AbandonedDetectionResponse, AbandonedDetectionExecutor, OutputImage


def build_response(context):
    outputImage = OutputImage(value=context.image)
    outputDetections = OutputDetections(value=context.detections)
    Outputs = AbandonedDetectionOutputs(outputImage=outputImage, outputDetections=outputDetections)
    packageResponse = AbandonedDetectionResponse(outputs=Outputs)
    packageExecutor = AbandonedDetectionExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
