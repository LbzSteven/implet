

selected_uni = [
    "ECG200",
    "Beef",
    "ElectricDevices",
    "Earthquakes",
    "Wafer",
    "PowerCons",
    "NonInvasiveFetalECGThorax1",
    "ECG5000",
    "GunPoint",
    "HandOutlines",
    "CBF",
    "FordA",
    "TwoPatterns",
    "UWaveGestureLibraryAll",
    "Chinatown",
    "Yoga",
    "DistalPhalanxOutlineCorrect",
    "Computers",
    "ShapesAll",
    "Strawberry"
]

xai_names = ['DeepLift',
             'IntegratedGradients',
             'InputXGradient',
             'Saliency',
             'Lime',
             'Occlusion',
             'KernelShap',
             'GuidedBackprop']

def create_1d_zigzag_array(size):
    zigzag_array = []
    for i in range(size):
        if i % 2 == 0:
            zigzag_array.append(1)  # Increasing for even indices
        else:
            zigzag_array.append(0)  # Decreasing for odd indices
    return zigzag_array