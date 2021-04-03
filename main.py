from utensil import get_logger
from utensil.random_search import UniformBetweenParam, ExponentialBetweenParam, RandomizedChoices, RandomizedDispatcher, \
    BooleanParam

from algorithm.kernel_svm import KernelSvmRandomSearch, SvmConfig, KernelChoices, PolyConfig, RbfConfig, SigmoidConfig
from algorithm.linear_svm import LinearSvmRandomSearch, LinearSvmConfig, PenaltyChoices, LossChoices

logger = get_logger(__name__)


poly_config_temp = PolyConfig(
    gamma=ExponentialBetweenParam(1e-6, 1e3, float),
    degree=UniformBetweenParam(2, 10, int),
)

rbf_config_temp = RbfConfig(
    gamma=ExponentialBetweenParam(1e-6, 1e3, float),
)

sigmoid_config_temp = SigmoidConfig(
    gamma=ExponentialBetweenParam(1e-6, 1e3, float),
)

svm_config_temp = SvmConfig(
    timeout=300,
    train_ratio=UniformBetweenParam(0.3, 0.6, float),

    c=ExponentialBetweenParam(1e-6, 1e3, float),
    kernel=RandomizedChoices(KernelChoices.POLY, KernelChoices.RBF, KernelChoices.SIGMOID),
    kernel_config=RandomizedDispatcher('kernel', {
        KernelChoices.POLY: poly_config_temp,
        KernelChoices.RBF: rbf_config_temp,
        KernelChoices.SIGMOID: sigmoid_config_temp
    })
)


kernel_svm_rsearch = KernelSvmRandomSearch(logger)
kernel_svm_rsearch.train('data/train.csv', 'data/test.csv', svm_config_temp, model_id_range=range(100))


linear_svm_config_temp = LinearSvmConfig(
    timeout=300,
    train_ratio=UniformBetweenParam(0.3, 0.6, float),

    c=ExponentialBetweenParam(1e-6, 1e3, float),
    dual=BooleanParam(),
    penalty=RandomizedChoices(PenaltyChoices.L1, PenaltyChoices.L2),
    loss=RandomizedChoices(LossChoices.HINGE, LossChoices.SQUARED_HINGE),
)

linear_svm_rsearch = LinearSvmRandomSearch(logger)
linear_svm_rsearch.train('data/train.csv', 'data/test.csv', linear_svm_config_temp, model_id_range=range(100))
