
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package,Detection, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class InputMaskShort(Input):
    name: Literal["inputMaskShort"] = "inputMaskShort"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Mask Short FG"


class InputMaskLong(Input):
    name: Literal["inputMaskLong"] = "inputMaskLong"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Mask Long FG"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"

class SSIMFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class SSIMFalseTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class TimeToStatic(Config):
    name: Literal["timeToStatic"] = "timeToStatic"
    value: float = Field(default=10.0, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Background Init Duration"


class ThresholdAlpha(Config):
    name: Literal["alpha"] = "alpha"
    value: float = Field(default=0.8, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Alpha"


class ThresholdBeta(Config):
    name: Literal["beta"] = "beta"
    value: float = Field(default=0.3, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Beta"

class SSIMThreshold(Config):
    name: Literal["ssimThreshold"] = "ssimThreshold"
    value: float = Field(default=0.8, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "SSIM Threshold"

class SSIMKernelSize(Config):
    name: Literal["ssimKernelSize"] = "ssimKernelSize"
    value: int = Field(default=12, ge=1, le=51)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "SSIM Kernel Size"

class WarningRatio(Config):
    name: Literal["warningRatio"] = "warningRatio"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Warning Ratio"

class SSIM(Config):
    name: Literal["ssim"] = "ssim",
    value: Union[SSIMFalse, SSIMFalseTrue]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

class AbandonedDetectionInputs(Inputs):
    inputImage: InputImage
    inputMaskShort: InputMaskShort
    inputMaskLong: InputMaskLong


class AbandonedDetectionConfigs(Configs):
    timeToStatic: TimeToStatic
    alpha: ThresholdAlpha
    beta: ThresholdBeta
    ssim: SSIM
    ssimThreshold: SSIMThreshold
    ssimKernelSize: SSIMKernelSize
    warningRatio: WarningRatio


class AbandonedDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class AbandonedDetectionRequest(Request):
    inputs: Optional[AbandonedDetectionInputs]
    configs: AbandonedDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class AbandonedDetectionResponse(Response):
    outputs: AbandonedDetectionOutputs



class AbandonedDetectionExecutor(Config):
    name: Literal["AbandonedDetection"] = "AbandonedDetection"
    value: Union[AbandonedDetectionRequest, AbandonedDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Abandoned Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[AbandonedDetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["AbandonedDetection"] = "AbandonedDetection"
