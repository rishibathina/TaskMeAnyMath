NUM_OPTIONS = 4

# MathQA

from geometry_task import *
from algebra_task import *

template_to_generator = {
        'point_slope_templates.json': PointSlopeGenerator,
        'linear_system_templates.json': LinearEquationSystemGenerator,
        'exponential_decay_templates.json': ExponentialDecayGenerator,
        'exponent_templates.json': SimpleExponentGenerator,
        'matrix_templates.json': MatrixArithmeticGenerator,
        'ratio_templates.json': RatioTaskGenerator,
        'basic_operation_templates.json': BasicArithmeticOperationsGenerator,
        'polynomial_factor_templates.json': PolynomialFactoringGenerator,
        'remainder_theorem_templates.json': RemainderTheoremGenerator,
        'quadratic_formula_templates.json': QuadraticFormulaGenerator,
        'triangle_area_templates.json': TriangleAreaGenerator,
        'volume_rectangular_prism_templates.json': RectangularPrismVolumeGenerator,
        'circle_templates.json': CircleGenerator,
        'perimeter_templates.json': TrianglePerimeterGenerator,
        'sideLength_template.json': SquareSideLengthGenerator,
        'midpoint_templates.json': MidpointGenerator,
        'cone_volume_templates.json': ConeVolumeGenerator,
        'angle_templates.json': AngleGenerator,
        'arcLength_template.json': ArcLengthGenerator,
        'point_distance_template.json': PointDistanceGenerator,
        'pythagoreanTheorem_template.json': PythagoreanTheoremGenerator,
        'sphere_volume_templates.json': SphereVolumeGenerator,
        'angle_sum_templates.json': AngleSumGenerator,
        'perpendicular_template.json': PerpendicularGenerator,
        'intersection_templates.json': IntersectionGenerator
    }

template_to_task_type = {
    'point_slope_templates.json': 'slope',
    'linear_system_templates.json': 'linear_system',
    'exponential_decay_templates.json': 'exponential_decay',
    'exponent_templates.json': 'exponent',
    'matrix_templates.json': 'matrix_arithmetic',
    'ratio_templates.json': 'ratio',
    'basic_operation_templates.json': 'basic_arithmetic_operations',
    'polynomial_factor_templates.json': 'polynomial_factor',
    'remainder_theorem_templates.json': 'remainder_theorem',
    'quadratic_formula_templates.json': 'quadratic_formula',
    'triangle_area_templates.json': 'triangle_area',
    'volume_rectangular_prism_templates.json': 'rectangular_prism_volume',
    'circle_templates.json': 'circle',
    'perimeter_templates.json': 'triangle_perimeter',
    'sideLength_template.json': 'square_side_length',
    'midpoint_templates.json': 'midpoint',
    'cone_volume_templates.json': 'cone_volume',
    'angle_templates.json': 'angle',
    'arcLength_template.json': 'arc_length',
    'point_distance_template.json': 'point_distance',
    'pythagoreanTheorem_template.json': 'pythagorean_theorem',
    'sphere_volume_templates.json': 'sphere_volume',
    'angle_sum_templates.json': 'angle_sum',
    'perpendicular_template.json': 'perpendicular',
    'intersection_templates.json': 'intersection',
}

# TextQA

MAX_OUTPUT_LEN_MC = 15
MAX_OUTPUT_LEN_ANSWER_ONLY = 50
TEMPERATURE_FOR_ANSWER_ONLY = 0.2
TOP_P_FOR_ANSWER_ONLY = 0.7

MAX_OUTPUT_LEN_REASONING = 400
TEMPERATURE_FOR_REASONING = 0.2
TOP_P_FOR_REASONING = 0.7
# try temperature .2 top_p .7

# ImageQA

IMAGE_H = 512
IMAGE_W = 512

# VideoQA

VIDEO_H = 224
VIDEO_W = 224
VIDEO_FPS = 4
VIDEO_NUM_FRAMES = 16
