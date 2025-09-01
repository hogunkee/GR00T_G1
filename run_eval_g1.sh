#!/bin/bash
TAG=$1
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlateToPanSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromTrayToPotSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PnPCupToDrawerClose_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PnPPotatoToMicrowaveClose_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PnPMilkToMicrowaveClose_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PnPBottleToCabinetClose_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PnPWineToCabinetClose_G1ArmsAndWaistDex31Hands_Env ${TAG} &&
./scripts/simul_eval_g1.sh 0 g1_unified/PnPCanToDrawerClose_G1ArmsAndWaistDex31Hands_Env ${TAG}
