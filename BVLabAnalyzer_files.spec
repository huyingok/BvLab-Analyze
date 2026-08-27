# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['BVLabAnalyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
('D:\\SY\\CellNeuralBloodVessel\\old_no_nnUnet_codes\\icon_images\\', 'icon_images\\'),
('D:\\SY\\CellNeuralBloodVessel\\old_no_nnUnet_codes\\DefaultModels\\', 'DefaultModels\\'),
('D:\\SY\\CellNeuralBloodVessel\\old_no_nnUnet_codes\\config\\', 'config\\'),
('D:\\SY\\CellNeuralBloodVessel\\old_no_nnUnet_codes\\bv\\', 'bv\\'),
('D:\\SY\\CellNeuralBloodVessel\\old_no_nnUnet_codes\\icon_image\\', 'icon_image\\'),
('D:\\SY\\CellNeuralBloodVessel\\old_no_nnUnet_codes\\BVExample\\', 'BVExample\\')],
    hiddenimports=['tensorboard', 'torch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    icon=['.\\icon_image\\icon.ico'],
    exclude_binaries=True,
    name='BVLabAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BVLabAnalyzer',
)
