"""MorphAlign: general-purpose channel-agnostic image registration GUI."""
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import cv2
import gradio as gr
from morphalign import read_image, preview_rgb, register, export_result

HERE=Path(__file__).resolve().parent

def _preview(array, max_side=850):
    rgb=preview_rgb(array)
    h,w=rgb.shape[:2]
    if max(h,w)>max_side:
        scale=max_side/max(h,w)
        rgb=cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    return rgb

def _parse_channels(value):
    value=str(value).strip()
    if not value or value.lower()=='auto':return 'auto'
    try:return [int(c.strip()) for c in value.split(',') if c.strip()]
    except ValueError as e: raise gr.Error('Channel indices must be comma-separated integers (or auto)') from e

def execute(image1_file,image2_file,channels,deformable,samples,rounds,max_shift):
    if image1_file is None or image2_file is None:
        raise gr.Error('Choose both image1 and image2')
    image1=read_image(image1_file)
    image2=read_image(image2_file)
    messages=[]
    result=register(image1,image2,channels=_parse_channels(channels),
        deformable=bool(deformable),sobol_samples=int(samples),
        exploration_rounds=int(rounds),max_displacement=float(max_shift),
        progress=messages.append)
    folder=Path(tempfile.mkdtemp(prefix='morphalign_'))
    export_result(result,image1,image2,folder)
    archive=folder/'results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in folder.iterdir():
            if file!=archive:z.write(file,arcname=file.name)
    info=(f'**image1:** {image1.shape}, {image1.dtype} · '
          f'**image2:** {image2.shape}, {image2.dtype} · '
          f'**registered:** {result["registered"].shape}, {result["registered"].dtype} '
          f'\n\n**Affine feature NCC:** {result["affine"]["final_ncc"]:.3f} · '
          f'**Reference coverage:** {result["affine"]["reference_coverage"]:.1%}'
          '\n\nFeature NCC is an internal optimization metric, not independently measured registration accuracy.')
    return (_preview(image1),_preview(image2),_preview(result['affine_registered']),
            _preview(result['registered']),info,str(archive))

with gr.Blocks(title='MorphAlign') as app:
    gr.Markdown('# MorphAlign\n**Explore globally. Align locally.** Two images in, one spatial alignment out. Any number of input channels; no stain labels required.')
    gr.Markdown('Image 1 is the **fixed/reference** image. Image 2 is the **moving** image, resampled onto image 1 coordinates. Supported: grayscale or H×W×C TIFF/PNG/JPEG. Example images are downsampled versions of the supplied microscopy pair.')
    with gr.Row():
        image1_file=gr.File(label='image1.tiff · reference',file_types=['.tif','.tiff','.png','.jpg','.jpeg'],type='filepath',value=str(HERE/'examples/image1.tiff'))
        image2_file=gr.File(label='image2.tiff · moving',file_types=['.tif','.tiff','.png','.jpg','.jpeg'],type='filepath',value=str(HERE/'examples/image2.tiff'))
    with gr.Accordion('Settings',open=True):
        channels=gr.Textbox(label='Channels used to estimate alignment (image2)',value='auto',info='auto selects all; or specify 0-based indices, e.g. 0,2,4. All original channels are always saved.')
        with gr.Row():
            deformable=gr.Checkbox(label='Local deformable refinement',value=True)
            samples=gr.Slider(64,1024,value=256,step=64,label='Global exploration samples')
            rounds=gr.Slider(0,5,value=2,step=1,label='Exploration rounds')
            max_shift=gr.Slider(5,80,value=30,step=5,label='Maximum local displacement, px')
    go=gr.Button('Register image2 to image1',variant='primary')
    with gr.Row():
        p1=gr.Image(label='Image 1 · reference',interactive=False)
        p2=gr.Image(label='Image 2 · original',interactive=False)
    with gr.Row():
        pa=gr.Image(label='Image 2 · affine aligned',interactive=False)
        pf=gr.Image(label='Image 2 · final aligned',interactive=False)
    summary=gr.Markdown()
    download=gr.File(label='Download registered TIFF and transforms')
    go.click(execute,inputs=[image1_file,image2_file,channels,deformable,samples,rounds,max_shift],outputs=[p1,p2,pa,pf,summary,download])
    gr.Markdown('**Research prototype:** Inspect correspondences and deformation fields. Arbitrary channel counts do not guarantee arbitrary contrast relationships can be aligned.')

if __name__=='__main__': app.launch()
