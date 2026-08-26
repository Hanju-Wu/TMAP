
% minFunc
fprintf('Compiling minFunc files...\n');
mex -compatibleArrayDims minFunc_2012/mex/mcholC.c -outdir minFunc_2012/compiled
mex -compatibleArrayDims minFunc_2012/mex/lbfgsC.c -outdir minFunc_2012/compiled
mex -compatibleArrayDims minFunc_2012/mex/lbfgsAddC.c -outdir minFunc_2012/compiled
mex -compatibleArrayDims minFunc_2012/mex/lbfgsProdC.c -outdir minFunc_2012/compiled

% UGM
fprintf('Compiling UGM files...\n');
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_makeEdgeVEC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_ExactC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_ExactC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_ChainC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_makeClampedPotentialsC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_ICMC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_GraphCutC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Sample_GibbsC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_MFC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_LBPC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_LBPC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_TRBPC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_TRBPC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_CRF_makePotentialsC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_CRF_PseudoNLLC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_LogConfigurationPotentialC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_AlphaExpansionC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_AlphaExpansionBetaShrinkC.c
mex -compatibleArrayDims -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_CRF_NLLC.c