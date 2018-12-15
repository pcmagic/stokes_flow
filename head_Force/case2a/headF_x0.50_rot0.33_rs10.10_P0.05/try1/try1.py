# state file generated using paraview version 5.5.0

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.5.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1003, 569]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.0007247328758239746, -0.5985283253830858, 0.008948802947998047]
renderView1.StereoType = 0
renderView1.CameraPosition = [0.0007247328758239746, 4.8562438354291455, 0.008948802947998047]
renderView1.CameraFocalPoint = [0.0007247328758239746, -0.5985283253830858, 0.008948802947998047]
renderView1.CameraViewUp = [1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 5.361304056495508
renderView1.Background = [0.32, 0.34, 0.43]
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'from paraview.simple import *
renderView1.AxesGrid.XTitleFontFile = ''
renderView1.AxesGrid.YTitleFontFile = ''
renderView1.AxesGrid.ZTitleFontFile = ''
renderView1.AxesGrid.XLabelFontFile = ''
renderView1.AxesGrid.YLabelFontFile = ''
renderView1.AxesGrid.ZLabelFontFile = ''

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Legacy VTK Reader'
headF_x000_rot200_rs1030_P050_vtkU_ = LegacyVTKReader(FileNames=['C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_001.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_002.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_003.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_004.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_005.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_006.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_007.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_008.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_009.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_010.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_011.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_012.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_013.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_014.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_015.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_016.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_017.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_018.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_019.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_020.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_021.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_022.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_023.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_024.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_025.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_026.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_027.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_028.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_029.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_030.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_031.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_032.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_033.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_034.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_035.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_036.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_037.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_038.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_039.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_040.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_041.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_042.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_043.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_044.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_045.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_046.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_047.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_048.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_049.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_050.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_051.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_052.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_053.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_054.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_055.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_056.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_057.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_058.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_059.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_060.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_061.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_062.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_063.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_064.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_065.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_066.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_067.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_068.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_069.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_070.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_071.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_072.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_073.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_074.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_075.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_076.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_077.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_078.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_079.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_080.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_081.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_082.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_083.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_084.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_085.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_086.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_087.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_088.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_089.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_090.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_091.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_092.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_093.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_094.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_095.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_096.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_097.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_098.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_099.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_100.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_101.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_102.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_103.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_104.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_105.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_106.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_107.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_108.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_109.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_110.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_111.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_112.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_113.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_114.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_115.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_116.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_117.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_118.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_119.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_120.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_121.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_122.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_123.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_124.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_125.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_126.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_127.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_128.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_129.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_130.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_131.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_132.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_133.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_134.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_135.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_136.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_137.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_138.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_139.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_140.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_141.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_142.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_143.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_144.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_145.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_146.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_147.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_148.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_149.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_150.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_151.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_152.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_153.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_154.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_155.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_156.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_157.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_158.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_159.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_160.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_161.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_162.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_163.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_164.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_165.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_166.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_167.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_168.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_169.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_170.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_171.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_172.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_173.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_174.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_175.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_176.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_177.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_178.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_179.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_180.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_181.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_182.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_183.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_184.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_185.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_186.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_187.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_188.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_189.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_190.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_191.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_192.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_193.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_194.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_195.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_196.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_197.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_198.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_199.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_200.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_201.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_202.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_203.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_204.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_205.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_206.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_207.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_208.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_209.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_210.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_211.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_212.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_213.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_214.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_215.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_216.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_217.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_218.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_219.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_220.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_221.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_222.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_223.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_224.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_225.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_226.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_227.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_228.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_229.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_230.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_231.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_232.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_233.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_234.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_235.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_236.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_237.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_238.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_239.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_240.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_241.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_242.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_243.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_244.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_245.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_246.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_247.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_248.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_249.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_250.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_251.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_252.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_253.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_254.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_255.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_256.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_257.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_258.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_259.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_260.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_261.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_262.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_263.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_264.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_265.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_266.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_267.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_268.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_269.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_270.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_271.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_272.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_273.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_274.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_275.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_276.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_277.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_278.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_279.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_280.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_281.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_282.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_283.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_284.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_285.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_286.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_287.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_288.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_289.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_290.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_291.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_292.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_293.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_294.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_295.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_296.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_297.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_298.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_299.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_300.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_301.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_302.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_303.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_304.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_305.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_306.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_307.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_308.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_309.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_310.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_311.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_312.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_313.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_314.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_315.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_316.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_317.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_318.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_319.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_320.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_321.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_322.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_323.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_324.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_325.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_326.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_327.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_328.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_329.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_330.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_331.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_332.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_333.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_334.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_335.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_336.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_337.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_338.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_339.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_340.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_341.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_342.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_343.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_344.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_345.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_346.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_347.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_348.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_349.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_350.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_351.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_352.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_353.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_354.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_355.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_356.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_357.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_358.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_359.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_360.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_361.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_362.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_363.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_364.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_365.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_366.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_367.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_368.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_369.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_370.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_371.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_372.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_373.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_374.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_375.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_376.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_377.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_378.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_379.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_380.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_381.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_382.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_383.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_384.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_385.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_386.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_387.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_388.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_389.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_390.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_391.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_392.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_393.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_394.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_395.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_396.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_397.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_398.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_399.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_400.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_401.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_402.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_403.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_404.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_405.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_406.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_407.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_408.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_409.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_410.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_411.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_412.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_413.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_414.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_415.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_416.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_417.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_418.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_419.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_420.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_421.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_422.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_423.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_424.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_425.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_426.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_427.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_428.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_429.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_430.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_431.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_432.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_433.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_434.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_435.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_436.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_437.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_438.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_439.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_440.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_441.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_442.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_443.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_444.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_445.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_446.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_447.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_448.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_449.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_450.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_451.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_452.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_453.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_454.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_455.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_456.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_457.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_458.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_459.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_460.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_461.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_462.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_463.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_464.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_465.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_466.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_467.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_468.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_469.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_470.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_471.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_472.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_473.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_474.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_475.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_476.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_477.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_478.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_479.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_480.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_481.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_482.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_483.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_484.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_485.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_486.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_487.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_488.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_489.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_490.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_491.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_492.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_493.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_494.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_495.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_496.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_497.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_498.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_499.vtk', 'C:\\Users\\pcmag\\OneDrive - cumt.edu.cn\\Microorgnisms\\codes\\stokes_flow\\head_Force\\case2a\\headF_x0.50_rot0.33_rs10.10_P0.05\\try1\\headF_x0.00_rot2.00_rs10.30_P0.50_vtkU_500.vtk'])

# create a new 'Glyph'
glyph1 = Glyph(Input=headF_x000_rot200_rs1030_P050_vtkU_,
    GlyphType='Arrow')
glyph1.Scalars = ['POINTS', 'u_Magnitude']
glyph1.Vectors = ['POINTS', 'u']
glyph1.ScaleFactor = 0.3
glyph1.MaximumNumberOfSamplePoints = 1000
glyph1.GlyphTransform = 'Transform2'

# create a new 'Extract Time Steps'
extractTimeSteps1 = ExtractTimeSteps(Input=headF_x000_rot200_rs1030_P050_vtkU_)
extractTimeSteps1.TimeStepIndices = [0, 0, 10, 20, 31, 41, 51, 61, 71, 81, 92, 102, 112, 122, 132, 143, 153, 163, 173, 183, 193, 204, 214, 224, 234, 244, 255, 265, 275, 285, 295, 306, 316, 326, 336, 346, 356, 367, 377, 387, 397, 407, 418, 428, 438, 448, 458, 468, 479, 489, 499]
extractTimeSteps1.TimeStepRange = [0, 499]

# create a new 'Clip'
clip1 = Clip(Input=glyph1)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'GlyphVector_Magnitude']
clip1.Value = 1.1289654604487519e-06

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.0007247328758239746, 0.0008057355880737305, 0.008159160614013672]
clip1.ClipType.Normal = [0.0, 1.0, 0.0]

# create a new 'Stream Tracer'
streamTracer1 = StreamTracer(Input=headF_x000_rot200_rs1030_P050_vtkU_,
    SeedType='Point Source')
streamTracer1.Vectors = ['POINTS', 'u']
streamTracer1.MaximumStreamlineLength = 10.0

# init the 'Point Source' selected for 'SeedType'
streamTracer1.SeedType.NumberOfPoints = 50

# create a new 'Tube'
tube1 = Tube(Input=streamTracer1)
tube1.Scalars = ['POINTS', 'u_Magnitude']
tube1.Vectors = ['POINTS', 'Normals']
tube1.Radius = 0.01

# create a new 'Annotate Time'
annotateTime1 = AnnotateTime()

# create a new 'Clip'
clip2 = Clip(Input=tube1)
clip2.ClipType = 'Plane'
clip2.Scalars = ['POINTS', 'AngularVelocity']
clip2.Value = -0.16358775164721084

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.06068718433380127, 0.00868251919746399, 0.0004286766052246094]
clip2.ClipType.Normal = [0.0, 1.0, 0.0]

# create a new 'Clip'
clip3 = Clip(Input=headF_x000_rot200_rs1030_P050_vtkU_)
clip3.ClipType = 'Plane'
clip3.Scalars = ['POINTS', 'u_Magnitude']
clip3.Value = 0.23967228178577227

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Normal = [0.0, 1.0, 0.0]

# create a new 'Contour'
contour1 = Contour(Input=clip3)
contour1.ContourBy = ['POINTS', 'u_Magnitude']
contour1.Isosurfaces = [0.004999999999999999, 0.008340502686000294, 0.01391279701103562, 0.023207944168063883, 0.03871318413405634, 0.0645774832507442, 0.10772173450159418, 0.17969068319023135, 0.2997421251594704, 0.5]
contour1.PointMergeMethod = 'Uniform Binning'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from headF_x000_rot200_rs1030_P050_vtkU_
headF_x000_rot200_rs1030_P050_vtkU_Display = Show(headF_x000_rot200_rs1030_P050_vtkU_, renderView1)

# trace defaults for the display properties.
headF_x000_rot200_rs1030_P050_vtkU_Display.Representation = 'Surface'
headF_x000_rot200_rs1030_P050_vtkU_Display.ColorArrayName = [None, '']
headF_x000_rot200_rs1030_P050_vtkU_Display.Opacity = 0.2
headF_x000_rot200_rs1030_P050_vtkU_Display.OSPRayScaleArray = 'u'
headF_x000_rot200_rs1030_P050_vtkU_Display.OSPRayScaleFunction = 'PiecewiseFunction'
headF_x000_rot200_rs1030_P050_vtkU_Display.SelectOrientationVectors = 'u'
headF_x000_rot200_rs1030_P050_vtkU_Display.SelectScaleArray = 'None'
headF_x000_rot200_rs1030_P050_vtkU_Display.GlyphType = 'Arrow'
headF_x000_rot200_rs1030_P050_vtkU_Display.GlyphTableIndexArray = 'None'
headF_x000_rot200_rs1030_P050_vtkU_Display.GaussianRadius = 0.05
headF_x000_rot200_rs1030_P050_vtkU_Display.SetScaleArray = ['POINTS', 'u']
headF_x000_rot200_rs1030_P050_vtkU_Display.ScaleTransferFunction = 'PiecewiseFunction'
headF_x000_rot200_rs1030_P050_vtkU_Display.OpacityArray = ['POINTS', 'u']
headF_x000_rot200_rs1030_P050_vtkU_Display.OpacityTransferFunction = 'PiecewiseFunction'
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid = 'GridAxesRepresentation'
headF_x000_rot200_rs1030_P050_vtkU_Display.SelectionCellLabelFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.SelectionPointLabelFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.PolarAxes = 'PolarAxesRepresentation'
headF_x000_rot200_rs1030_P050_vtkU_Display.ScalarOpacityUnitDistance = 0.3273541413744618

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
headF_x000_rot200_rs1030_P050_vtkU_Display.ScaleTransferFunction.Points = [-0.0594670210334, 0.0, 0.5, 0.0, 0.19008899476699997, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
headF_x000_rot200_rs1030_P050_vtkU_Display.OpacityTransferFunction.Points = [-0.0594670210334, 0.0, 0.5, 0.0, 0.19008899476699997, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid.XTitleFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid.YTitleFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid.ZTitleFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid.XLabelFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid.YLabelFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
headF_x000_rot200_rs1030_P050_vtkU_Display.PolarAxes.PolarAxisTitleFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.PolarAxes.PolarAxisLabelFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.PolarAxes.LastRadialAxisTextFontFile = ''
headF_x000_rot200_rs1030_P050_vtkU_Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from glyph1
glyph1Display = Show(glyph1, renderView1)

# get color transfer function/color map for 'GlyphVector'
glyphVectorLUT = GetColorTransferFunction('GlyphVector')
glyphVectorLUT.AutomaticRescaleRangeMode = 'Never'
glyphVectorLUT.RGBPoints = [0.004999999999999999, 0.231373, 0.298039, 0.752941, 0.04999999999999994, 0.865003, 0.865003, 0.865003, 0.49999999999999994, 0.705882, 0.0156863, 0.14902]
glyphVectorLUT.UseLogScale = 1
glyphVectorLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = ['POINTS', 'GlyphVector']
glyph1Display.LookupTable = glyphVectorLUT
glyph1Display.OSPRayScaleArray = 'GlyphVector'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'GlyphVector'
glyph1Display.ScaleFactor = 1.191840696334839
glyph1Display.SelectScaleArray = 'GlyphVector'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'GlyphVector'
glyph1Display.GaussianRadius = 0.05959203481674195
glyph1Display.SetScaleArray = ['POINTS', 'GlyphVector']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = ['POINTS', 'GlyphVector']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.SelectionCellLabelFontFile = ''
glyph1Display.SelectionPointLabelFontFile = ''
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [-0.04575273022055626, 0.0, 0.5, 0.0, 0.07327467948198318, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [-0.04575273022055626, 0.0, 0.5, 0.0, 0.07327467948198318, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
glyph1Display.DataAxesGrid.XTitleFontFile = ''
glyph1Display.DataAxesGrid.YTitleFontFile = ''
glyph1Display.DataAxesGrid.ZTitleFontFile = ''
glyph1Display.DataAxesGrid.XLabelFontFile = ''
glyph1Display.DataAxesGrid.YLabelFontFile = ''
glyph1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
glyph1Display.PolarAxes.PolarAxisTitleFontFile = ''
glyph1Display.PolarAxes.PolarAxisLabelFontFile = ''
glyph1Display.PolarAxes.LastRadialAxisTextFontFile = ''
glyph1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from clip1
clip1Display = Show(clip1, renderView1)

# get opacity transfer function/opacity map for 'GlyphVector'
glyphVectorPWF = GetOpacityTransferFunction('GlyphVector')
glyphVectorPWF.Points = [0.004999999999999999, 0.0, 0.5, 0.0, 0.49999999999999994, 1.0, 0.5, 0.0]
glyphVectorPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'GlyphVector']
clip1Display.LookupTable = glyphVectorLUT
clip1Display.OSPRayScaleArray = 'GlyphVector'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'GlyphVector'
clip1Display.ScaleFactor = 1.0382317066192628
clip1Display.SelectScaleArray = 'GlyphVector'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'GlyphVector'
clip1Display.GaussianRadius = 0.051911585330963135
clip1Display.SetScaleArray = ['POINTS', 'GlyphVector']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'GlyphVector']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.SelectionCellLabelFontFile = ''
clip1Display.SelectionPointLabelFontFile = ''
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = glyphVectorPWF
clip1Display.ScalarOpacityUnitDistance = 0.3086019300583357

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-0.050146378576755524, 0.0, 0.5, 0.0, 0.2415119707584381, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-0.050146378576755524, 0.0, 0.5, 0.0, 0.2415119707584381, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
clip1Display.DataAxesGrid.XTitleFontFile = ''
clip1Display.DataAxesGrid.YTitleFontFile = ''
clip1Display.DataAxesGrid.ZTitleFontFile = ''
clip1Display.DataAxesGrid.XLabelFontFile = ''
clip1Display.DataAxesGrid.YLabelFontFile = ''
clip1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
clip1Display.PolarAxes.PolarAxisTitleFontFile = ''
clip1Display.PolarAxes.PolarAxisLabelFontFile = ''
clip1Display.PolarAxes.LastRadialAxisTextFontFile = ''
clip1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from streamTracer1
streamTracer1Display = Show(streamTracer1, renderView1)

# get color transfer function/color map for 'Vorticity'
vorticityLUT = GetColorTransferFunction('Vorticity')
vorticityLUT.AutomaticRescaleRangeMode = 'Never'
vorticityLUT.RGBPoints = [0.04000000000000001, 0.231373, 0.298039, 0.752941, 0.3999999999999998, 0.865003, 0.865003, 0.865003, 4.000000000000001, 0.705882, 0.0156863, 0.14902]
vorticityLUT.UseLogScale = 1
vorticityLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
streamTracer1Display.Representation = 'Surface'
streamTracer1Display.ColorArrayName = ['POINTS', 'Vorticity']
streamTracer1Display.LookupTable = vorticityLUT
streamTracer1Display.OSPRayScaleArray = 'AngularVelocity'
streamTracer1Display.OSPRayScaleFunction = 'PiecewiseFunction'
streamTracer1Display.SelectOrientationVectors = 'Normals'
streamTracer1Display.ScaleFactor = 1.000011157989502
streamTracer1Display.SelectScaleArray = 'AngularVelocity'
streamTracer1Display.GlyphType = 'Arrow'
streamTracer1Display.GlyphTableIndexArray = 'AngularVelocity'
streamTracer1Display.GaussianRadius = 0.0500005578994751
streamTracer1Display.SetScaleArray = ['POINTS', 'AngularVelocity']
streamTracer1Display.ScaleTransferFunction = 'PiecewiseFunction'
streamTracer1Display.OpacityArray = ['POINTS', 'AngularVelocity']
streamTracer1Display.OpacityTransferFunction = 'PiecewiseFunction'
streamTracer1Display.DataAxesGrid = 'GridAxesRepresentation'
streamTracer1Display.SelectionCellLabelFontFile = ''
streamTracer1Display.SelectionPointLabelFontFile = ''
streamTracer1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamTracer1Display.ScaleTransferFunction.Points = [-2.0910175811158886, 0.0, 0.5, 0.0, 2.7400958520299055, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamTracer1Display.OpacityTransferFunction.Points = [-2.0910175811158886, 0.0, 0.5, 0.0, 2.7400958520299055, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
streamTracer1Display.DataAxesGrid.XTitleFontFile = ''
streamTracer1Display.DataAxesGrid.YTitleFontFile = ''
streamTracer1Display.DataAxesGrid.ZTitleFontFile = ''
streamTracer1Display.DataAxesGrid.XLabelFontFile = ''
streamTracer1Display.DataAxesGrid.YLabelFontFile = ''
streamTracer1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
streamTracer1Display.PolarAxes.PolarAxisTitleFontFile = ''
streamTracer1Display.PolarAxes.PolarAxisLabelFontFile = ''
streamTracer1Display.PolarAxes.LastRadialAxisTextFontFile = ''
streamTracer1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from tube1
tube1Display = Show(tube1, renderView1)

# get color transfer function/color map for 'u'
uLUT = GetColorTransferFunction('u')
uLUT.AutomaticRescaleRangeMode = 'Never'
uLUT.RGBPoints = [0.004999999999999999, 0.231373, 0.298039, 0.752941, 0.049999999999999996, 0.865003, 0.865003, 0.865003, 0.49999999999999994, 0.705882, 0.0156863, 0.14902]
uLUT.UseLogScale = 1
uLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
tube1Display.Representation = 'Surface'
tube1Display.ColorArrayName = ['POINTS', 'u']
tube1Display.LookupTable = uLUT
tube1Display.OSPRayScaleArray = 'AngularVelocity'
tube1Display.OSPRayScaleFunction = 'PiecewiseFunction'
tube1Display.SelectOrientationVectors = 'Normals'
tube1Display.ScaleFactor = 0.9999301433563232
tube1Display.SelectScaleArray = 'AngularVelocity'
tube1Display.GlyphType = 'Arrow'
tube1Display.GlyphTableIndexArray = 'AngularVelocity'
tube1Display.GaussianRadius = 0.049996507167816166
tube1Display.SetScaleArray = ['POINTS', 'AngularVelocity']
tube1Display.ScaleTransferFunction = 'PiecewiseFunction'
tube1Display.OpacityArray = ['POINTS', 'AngularVelocity']
tube1Display.OpacityTransferFunction = 'PiecewiseFunction'
tube1Display.DataAxesGrid = 'GridAxesRepresentation'
tube1Display.SelectionCellLabelFontFile = ''
tube1Display.SelectionPointLabelFontFile = ''
tube1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tube1Display.ScaleTransferFunction.Points = [-3.5176956847359735, 0.0, 0.5, 0.0, 2.918746461505828, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tube1Display.OpacityTransferFunction.Points = [-3.5176956847359735, 0.0, 0.5, 0.0, 2.918746461505828, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
tube1Display.DataAxesGrid.XTitleFontFile = ''
tube1Display.DataAxesGrid.YTitleFontFile = ''
tube1Display.DataAxesGrid.ZTitleFontFile = ''
tube1Display.DataAxesGrid.XLabelFontFile = ''
tube1Display.DataAxesGrid.YLabelFontFile = ''
tube1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
tube1Display.PolarAxes.PolarAxisTitleFontFile = ''
tube1Display.PolarAxes.PolarAxisLabelFontFile = ''
tube1Display.PolarAxes.LastRadialAxisTextFontFile = ''
tube1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from clip2
clip2Display = Show(clip2, renderView1)

# get opacity transfer function/opacity map for 'u'
uPWF = GetOpacityTransferFunction('u')
uPWF.Points = [0.004999999999999999, 0.0, 0.5, 0.0, 0.49999999999999994, 1.0, 0.5, 0.0]
uPWF.UseLogScale = 1
uPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'u']
clip2Display.LookupTable = uLUT
clip2Display.OSPRayScaleArray = 'AngularVelocity'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'Normals'
clip2Display.ScaleFactor = 0.9999263763427735
clip2Display.SelectScaleArray = 'AngularVelocity'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'AngularVelocity'
clip2Display.GaussianRadius = 0.049996318817138674
clip2Display.SetScaleArray = ['POINTS', 'AngularVelocity']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = ['POINTS', 'AngularVelocity']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.SelectionCellLabelFontFile = ''
clip2Display.SelectionPointLabelFontFile = ''
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = uPWF
clip2Display.ScalarOpacityUnitDistance = 0.2066207818647344

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip2Display.ScaleTransferFunction.Points = [-3.3963331830428247, 0.0, 0.5, 0.0, 2.5790176079320437, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip2Display.OpacityTransferFunction.Points = [-3.3963331830428247, 0.0, 0.5, 0.0, 2.5790176079320437, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
clip2Display.DataAxesGrid.XTitleFontFile = ''
clip2Display.DataAxesGrid.YTitleFontFile = ''
clip2Display.DataAxesGrid.ZTitleFontFile = ''
clip2Display.DataAxesGrid.XLabelFontFile = ''
clip2Display.DataAxesGrid.YLabelFontFile = ''
clip2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
clip2Display.PolarAxes.PolarAxisTitleFontFile = ''
clip2Display.PolarAxes.PolarAxisLabelFontFile = ''
clip2Display.PolarAxes.LastRadialAxisTextFontFile = ''
clip2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from clip3
clip3Display = Show(clip3, renderView1)

# trace defaults for the display properties.
clip3Display.Representation = 'Surface'
clip3Display.ColorArrayName = [None, '']
clip3Display.OSPRayScaleArray = 'u'
clip3Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip3Display.SelectOrientationVectors = 'u'
clip3Display.SelectScaleArray = 'None'
clip3Display.GlyphType = 'Arrow'
clip3Display.GlyphTableIndexArray = 'None'
clip3Display.GaussianRadius = 0.05
clip3Display.SetScaleArray = ['POINTS', 'u']
clip3Display.ScaleTransferFunction = 'PiecewiseFunction'
clip3Display.OpacityArray = ['POINTS', 'u']
clip3Display.OpacityTransferFunction = 'PiecewiseFunction'
clip3Display.DataAxesGrid = 'GridAxesRepresentation'
clip3Display.SelectionCellLabelFontFile = ''
clip3Display.SelectionPointLabelFontFile = ''
clip3Display.PolarAxes = 'PolarAxesRepresentation'
clip3Display.ScalarOpacityUnitDistance = 0.3906113473445861

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip3Display.ScaleTransferFunction.Points = [-0.04436862274374304, 0.0, 0.5, 0.0, 0.283981342291, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip3Display.OpacityTransferFunction.Points = [-0.04436862274374304, 0.0, 0.5, 0.0, 0.283981342291, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
clip3Display.DataAxesGrid.XTitleFontFile = ''
clip3Display.DataAxesGrid.YTitleFontFile = ''
clip3Display.DataAxesGrid.ZTitleFontFile = ''
clip3Display.DataAxesGrid.XLabelFontFile = ''
clip3Display.DataAxesGrid.YLabelFontFile = ''
clip3Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
clip3Display.PolarAxes.PolarAxisTitleFontFile = ''
clip3Display.PolarAxes.PolarAxisLabelFontFile = ''
clip3Display.PolarAxes.LastRadialAxisTextFontFile = ''
clip3Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from contour1
contour1Display = Show(contour1, renderView1)

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', 'u']
contour1Display.LookupTable = uLUT
contour1Display.OSPRayScaleArray = 'Normals'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.1323624347903103
contour1Display.SelectScaleArray = 'None'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'None'
contour1Display.GaussianRadius = 0.006618121739515515
contour1Display.SetScaleArray = ['POINTS', 'Normals']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'Normals']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.SelectionCellLabelFontFile = ''
contour1Display.SelectionPointLabelFontFile = ''
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [-0.9999999403953552, 0.0, 0.5, 0.0, 0.9999940395355225, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [-0.9999999403953552, 0.0, 0.5, 0.0, 0.9999940395355225, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
contour1Display.DataAxesGrid.XTitleFontFile = ''
contour1Display.DataAxesGrid.YTitleFontFile = ''
contour1Display.DataAxesGrid.ZTitleFontFile = ''
contour1Display.DataAxesGrid.XLabelFontFile = ''
contour1Display.DataAxesGrid.YLabelFontFile = ''
contour1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
contour1Display.PolarAxes.PolarAxisTitleFontFile = ''
contour1Display.PolarAxes.PolarAxisLabelFontFile = ''
contour1Display.PolarAxes.LastRadialAxisTextFontFile = ''
contour1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from annotateTime1
annotateTime1Display = Show(annotateTime1, renderView1)

# trace defaults for the display properties.
annotateTime1Display.FontFile = ''

# show data from extractTimeSteps1
extractTimeSteps1Display = Show(extractTimeSteps1, renderView1)

# trace defaults for the display properties.
extractTimeSteps1Display.Representation = 'Surface'
extractTimeSteps1Display.ColorArrayName = [None, '']
extractTimeSteps1Display.OSPRayScaleArray = 'u'
extractTimeSteps1Display.OSPRayScaleFunction = 'PiecewiseFunction'
extractTimeSteps1Display.SelectOrientationVectors = 'u'
extractTimeSteps1Display.SelectScaleArray = 'None'
extractTimeSteps1Display.GlyphType = 'Arrow'
extractTimeSteps1Display.GlyphTableIndexArray = 'None'
extractTimeSteps1Display.GaussianRadius = 0.05
extractTimeSteps1Display.SetScaleArray = ['POINTS', 'u']
extractTimeSteps1Display.ScaleTransferFunction = 'PiecewiseFunction'
extractTimeSteps1Display.OpacityArray = ['POINTS', 'u']
extractTimeSteps1Display.OpacityTransferFunction = 'PiecewiseFunction'
extractTimeSteps1Display.DataAxesGrid = 'GridAxesRepresentation'
extractTimeSteps1Display.SelectionCellLabelFontFile = ''
extractTimeSteps1Display.SelectionPointLabelFontFile = ''
extractTimeSteps1Display.PolarAxes = 'PolarAxesRepresentation'
extractTimeSteps1Display.ScalarOpacityUnitDistance = 0.3273541413744618

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
extractTimeSteps1Display.ScaleTransferFunction.Points = [-0.0446706807521, 0.0, 0.5, 0.0, 0.283981342291, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
extractTimeSteps1Display.OpacityTransferFunction.Points = [-0.0446706807521, 0.0, 0.5, 0.0, 0.283981342291, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
extractTimeSteps1Display.DataAxesGrid.XTitleFontFile = ''
extractTimeSteps1Display.DataAxesGrid.YTitleFontFile = ''
extractTimeSteps1Display.DataAxesGrid.ZTitleFontFile = ''
extractTimeSteps1Display.DataAxesGrid.XLabelFontFile = ''
extractTimeSteps1Display.DataAxesGrid.YLabelFontFile = ''
extractTimeSteps1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
extractTimeSteps1Display.PolarAxes.PolarAxisTitleFontFile = ''
extractTimeSteps1Display.PolarAxes.PolarAxisLabelFontFile = ''
extractTimeSteps1Display.PolarAxes.LastRadialAxisTextFontFile = ''
extractTimeSteps1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# setup the color legend parameters for each legend in this view

# get color legend/bar for glyphVectorLUT in view renderView1
glyphVectorLUTColorBar = GetScalarBar(glyphVectorLUT, renderView1)
glyphVectorLUTColorBar.Orientation = 'Horizontal'
glyphVectorLUTColorBar.WindowLocation = 'AnyLocation'
glyphVectorLUTColorBar.Position = [0.10145563310069769, 0.02987697715289983]
glyphVectorLUTColorBar.Title = 'GlyphVector'
glyphVectorLUTColorBar.ComponentTitle = 'Magnitude'
glyphVectorLUTColorBar.TitleFontFile = ''
glyphVectorLUTColorBar.LabelFontFile = ''
glyphVectorLUTColorBar.ScalarBarLength = 0.3300000000000004

# set color bar visibility
glyphVectorLUTColorBar.Visibility = 1

# get color legend/bar for uLUT in view renderView1
uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Orientation = 'Horizontal'
uLUTColorBar.WindowLocation = 'AnyLocation'
uLUTColorBar.Position = [0.586001994017946, 0.0210896309314587]
uLUTColorBar.Title = 'u'
uLUTColorBar.ComponentTitle = 'Magnitude'
uLUTColorBar.TitleFontFile = ''
uLUTColorBar.LabelFontFile = ''
uLUTColorBar.ScalarBarLength = 0.3300000000000003

# set color bar visibility
uLUTColorBar.Visibility = 0

# get color transfer function/color map for 'Rotation'
rotationLUT = GetColorTransferFunction('Rotation')
rotationLUT.RGBPoints = [-16.07484915968382, 0.231373, 0.298039, 0.752941, 352.7289601316403, 0.865003, 0.865003, 0.865003, 721.5327694229644, 0.705882, 0.0156863, 0.14902]
rotationLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for rotationLUT in view renderView1
rotationLUTColorBar = GetScalarBar(rotationLUT, renderView1)
rotationLUTColorBar.Title = 'Rotation'
rotationLUTColorBar.ComponentTitle = ''
rotationLUTColorBar.TitleFontFile = ''
rotationLUTColorBar.LabelFontFile = ''

# set color bar visibility
rotationLUTColorBar.Visibility = 0

# get color legend/bar for vorticityLUT in view renderView1
vorticityLUTColorBar = GetScalarBar(vorticityLUT, renderView1)
vorticityLUTColorBar.Orientation = 'Horizontal'
vorticityLUTColorBar.WindowLocation = 'AnyLocation'
vorticityLUTColorBar.Position = [0.5740378863409771, 0.02460456942003514]
vorticityLUTColorBar.Title = 'Vorticity'
vorticityLUTColorBar.ComponentTitle = 'Magnitude'
vorticityLUTColorBar.TitleFontFile = ''
vorticityLUTColorBar.LabelFontFile = ''
vorticityLUTColorBar.ScalarBarLength = 0.3300000000000004

# set color bar visibility
vorticityLUTColorBar.Visibility = 0

# get color transfer function/color map for 'Normals'
normalsLUT = GetColorTransferFunction('Normals')
normalsLUT.RGBPoints = [0.9999999999999992, 0.231373, 0.298039, 0.752941, 1.0001220703124996, 0.865003, 0.865003, 0.865003, 1.000244140625, 0.705882, 0.0156863, 0.14902]
normalsLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for normalsLUT in view renderView1
normalsLUTColorBar = GetScalarBar(normalsLUT, renderView1)
normalsLUTColorBar.Title = 'Normals'
normalsLUTColorBar.ComponentTitle = 'Magnitude'
normalsLUTColorBar.TitleFontFile = ''
normalsLUTColorBar.LabelFontFile = ''

# set color bar visibility
normalsLUTColorBar.Visibility = 0

# get color transfer function/color map for 'AngularVelocity'
angularVelocityLUT = GetColorTransferFunction('AngularVelocity')
angularVelocityLUT.RGBPoints = [-1.8249814183699717, 0.231373, 0.298039, 0.752941, -0.2589156039665166, 0.865003, 0.865003, 0.865003, 1.3071502104369386, 0.705882, 0.0156863, 0.14902]
angularVelocityLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for angularVelocityLUT in view renderView1
angularVelocityLUTColorBar = GetScalarBar(angularVelocityLUT, renderView1)
angularVelocityLUTColorBar.Title = 'AngularVelocity'
angularVelocityLUTColorBar.ComponentTitle = ''
angularVelocityLUTColorBar.TitleFontFile = ''
angularVelocityLUTColorBar.LabelFontFile = ''

# set color bar visibility
angularVelocityLUTColorBar.Visibility = 0

# get color transfer function/color map for 'u_Magnitude'
u_MagnitudeLUT = GetColorTransferFunction('u_Magnitude')
u_MagnitudeLUT.RGBPoints = [0.0049999999999999975, 0.231373, 0.298039, 0.752941, 0.15237106257973523, 0.865003, 0.865003, 0.865003, 0.29974212515947046, 0.705882, 0.0156863, 0.14902]
u_MagnitudeLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for u_MagnitudeLUT in view renderView1
u_MagnitudeLUTColorBar = GetScalarBar(u_MagnitudeLUT, renderView1)
u_MagnitudeLUTColorBar.Title = 'u_Magnitude'
u_MagnitudeLUTColorBar.ComponentTitle = ''
u_MagnitudeLUTColorBar.TitleFontFile = ''
u_MagnitudeLUTColorBar.LabelFontFile = ''

# set color bar visibility
u_MagnitudeLUTColorBar.Visibility = 0

# hide data in view
Hide(headF_x000_rot200_rs1030_P050_vtkU_, renderView1)

# show color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(glyph1, renderView1)

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(streamTracer1, renderView1)

# hide data in view
Hide(tube1, renderView1)

# hide data in view
Hide(clip2, renderView1)

# hide data in view
Hide(clip3, renderView1)

# hide data in view
Hide(contour1, renderView1)

# hide data in view
Hide(extractTimeSteps1, renderView1)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'u_Magnitude'
u_MagnitudePWF = GetOpacityTransferFunction('u_Magnitude')
u_MagnitudePWF.Points = [0.0049999999999999975, 0.0, 0.5, 0.0, 0.29974212515947046, 1.0, 0.5, 0.0]
u_MagnitudePWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'Normals'
normalsPWF = GetOpacityTransferFunction('Normals')
normalsPWF.Points = [0.9999999999999992, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
normalsPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'Vorticity'
vorticityPWF = GetOpacityTransferFunction('Vorticity')
vorticityPWF.Points = [0.04, 0.0, 0.5, 0.0, 4.0, 1.0, 0.5, 0.0]
vorticityPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'Rotation'
rotationPWF = GetOpacityTransferFunction('Rotation')
rotationPWF.Points = [-16.07484915968382, 0.0, 0.5, 0.0, 721.5327694229644, 1.0, 0.5, 0.0]
rotationPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'AngularVelocity'
angularVelocityPWF = GetOpacityTransferFunction('AngularVelocity')
angularVelocityPWF.Points = [-1.8249814183699717, 0.0, 0.5, 0.0, 1.3071502104369386, 1.0, 0.5, 0.0]
angularVelocityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(glyph1)
# ----------------------------------------------------------------