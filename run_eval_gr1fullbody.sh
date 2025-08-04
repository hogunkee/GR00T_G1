#!/bin/bash
TAG=$1
./scripts/simul_eval.sh 0 gr1_unified/PnPCupToDrawerClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PnPPotatoToMicrowaveClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PnPMilkToMicrowaveClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PnPBottleToCabinetClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PnPWineToCabinetClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PnPCanToDrawerClose_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1FullFourierHands_Env ${TAG} &&
./scripts/simul_eval.sh 0 gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1FullFourierHands_Env ${TAG}
