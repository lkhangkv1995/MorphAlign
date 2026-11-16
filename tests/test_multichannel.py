import cv2
import numpy as np
from morphalign.affine import _multichannel_structure, _generic_tissue_mask, preview_rgb, warp_af_to_he
from morphalign.pipeline import run_registration, export_result
from morphalign import register


def make_pair(shape, channels):
    h,w=shape
    rng=np.random.default_rng(7)
    pattern=rng.random(shape).astype(np.float32)
    pattern=cv2.GaussianBlur(pattern,(0,0),sigmaX=2.5)
    rgb=np.clip(255*np.stack([pattern,pattern**.7,1-pattern*.7],-1),0,255).astype(np.uint8)
    multi=np.stack([np.clip(pattern*(j+1)/(channels+1),0,1) for j in range(channels)],-1)
    return rgb,(multi*65535).astype(np.uint16)



def test_shapes_and_channels(tmp_path):
    ref,mov=make_pair((128,144),4)
    assert ref.shape==(128,144,3) and mov.shape==(128,144,4)
    assert _multichannel_structure(mov).shape==(1,128,144)
    assert _generic_tissue_mask(mov).shape==(128,144)
    assert preview_rgb(mov).shape==(128,144,3)
    result=run_registration(ref,mov,feature_mode='multichannel',deformable=False,
                            sobol_samples=16,exploration_rounds=0,affine_iterations=(3,3,3),progress=None)
    assert result['registered'].shape==mov.shape
    assert result['registered'].dtype==mov.dtype
    export_result(result,ref,mov,tmp_path)
    from tifffile import imread
    reloaded=imread(tmp_path/'registered_moving.tif')
    assert reloaded.shape==mov.shape and reloaded.dtype==np.uint16
    assert (tmp_path/'reference_to_moving_affine.npy').exists()


def test_deformation_preserves_all_channels():
    ref,mov=make_pair((120,150),4)
    out=run_registration(ref,mov,feature_mode='multichannel',deformable=True,
                         moving_channels=[0,2],sobol_samples=16,
                         exploration_rounds=0,affine_iterations=(2,2,2),
                         deform_iterations=(2,2),max_displacement=10,progress=None)
    assert out['registered'].shape==(120,150,4)
    assert out['registered'].dtype==np.uint16
    assert out['deformable'] is not None


def test_public_api_without_channel_assumptions():
    ref,mov=make_pair((120,150),5)
    result=register(ref,mov,channels='auto',deformable=False,
        sobol_samples=16,exploration_rounds=0,affine_iterations=(2,2,2),progress=None)
    assert result['registered'].shape==(120,150,5)
    assert result['registered'].dtype==np.uint16

def test_real_example_files_are_generic():
    from pathlib import Path
    from morphalign import read_image
    folder=Path(__file__).resolve().parents[1]/'examples'
    ref=read_image(folder/'image1.tiff')
    mov=read_image(folder/'image2.tiff')
    assert ref.ndim in (2,3) and mov.ndim in (2,3)
    assert ref.shape[:2]!=mov.shape[:2]
