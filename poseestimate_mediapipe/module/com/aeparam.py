class AEParam:
    COM_RATIO = {
        'M': {
            'head': 0.821,
            'body': 0.493,
            'upper_arm': 0.529,
            'fore_arm': 0.415,
            'hand': 0.891,
            'thigh': 0.475,
            'crus': 0.406,
            'foot': 0.595,
            'upper_torso': 0.428,
            'lower_torso': 0.609
        },
        'F': {
            'head': 0.759,
            'body': 0.506,
            'upper_arm': 0.523,
            'fore_arm': 0.423,
            'hand': 0.908,
            'thigh': 0.458,
            'crus': 0.410,
            'foot': 0.594,
            'upper_torso': 0.438,
            'lower_torso': 0.597
        }
    }

    PARTMASS_A0 = {
        'M': {
            'head': -1.1968,
            'body': -10.1647,
            'upper_arm': -0.36785,
            'fore_arm': -0.43807,
            'hand': -0.01474,
            'thigh': -4.53542,
            'crus': -1.71524,
            'foot': -0.26784,
            'upper_torso': -9.63322,
            'lower_torso': -5.70449
        },
        'F': {
            'head': -0.67895,
            'body': -14.61,
            'upper_arm': -0.49429,
            'fore_arm': -0.33838,
            'hand': -0.04586,
            'thigh': -3.74942,
            'crus': -1.40459,
            'foot': -0.39879,
            'upper_torso': -6.14877,
            'lower_torso': -6.87637
        }
    }

    PARTMASS_A1 = {
        'M': {
            'head': 25.9526,
            'body': 18.7503,
            'upper_arm': 1.15588,
            'fore_arm': 2.22923,
            'hand': 2.09424,
            'thigh': 14.5253,
            'crus': 6.04396,
            'foot': 2.61804,
            'upper_torso': 31.431,
            'lower_torso': 29.2113
        },
        'F': {
            'head': 22.8598,
            'body': 34.646,
            'upper_arm': 2.04431,
            'fore_arm': 2.42059,
            'hand': 1.68034,
            'thigh': 8.19775,
            'crus': 5.11653,
            'foot': 3.20138,
            'upper_torso': 26.8538,
            'lower_torso': 39.0758
        }
    }

    PARTMASS_A2 = {
        'M': {
            'head': 0.02604,
            'body': 0.48275,
            'upper_arm': 0.02712,
            'fore_arm': 0.01397,
            'hand': 0.00414,
            'thigh': 0.09324,
            'crus': 0.03885,
            'foot': 0.00545,
            'upper_torso': 0.27893,
            'lower_torso': 0.17966
        },
        'F': {
            'head': 0.02111,
            'body': 0.39995,
            'upper_arm': 0.02414,
            'fore_arm': 0.01079,
            'hand': 0.00478,
            'thigh': 0.13576,
            'crus': 0.04346,
            'foot': 0.00477,
            'upper_torso': 0.22507,
            'lower_torso': 0.17224
        }
    }

    # 慣性モーメント推定 (阿江, 1996)。
    # I[kg*m^2] = (A0 + A1 * composite_mass[kg] + A2 * segment_length[cm]) / 10000
    # composite_mass は体重 + 負荷(バーベル等)の合計質量。
    # thigh(大腿) / crus(下腿) のみ係数が公表されているため、他セグメントは未対応。
    # 移植元: squat_program/python/index_estimation/Joint_Torque_Estimation/parameter.py
    MOMENT_OF_INERTIA_A0 = {
        'M': {'thigh': -2043.38, 'crus': -1174.66},
        'F': {'thigh': -1851.78, 'crus': -830.815},
    }
    MOMENT_OF_INERTIA_A1 = {
        'M': {'thigh': 5547.75, 'crus': 3048.1},
        'F': {'thigh': 4347.35, 'crus': 2342.82},
    }
    MOMENT_OF_INERTIA_A2 = {
        'M': {'thigh': 10.6498, 'crus': 5.19169},
        'F': {'thigh': 17.6609, 'crus': 4.75943},
    }
