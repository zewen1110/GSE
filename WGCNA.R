# ===== 0. 加载包 =====
library(WGCNA)
options(stringsAsFactors = FALSE)
allowWGCNAThreads()

# ===== 1. 读入表达数据（行为基因，列为样本） =====
expr_raw <- read.csv("GSE96804_expr_matrix.csv", row.names = 1)
datExpr0 <- t(expr_raw)  # 转置，行为样本，列为基因

# ===== 2. 筛选方差前25%高变基因（更具生物信息量） =====
gene_var <- apply(datExpr0, 2, var, na.rm = TRUE)
datExpr1 <- datExpr0[, gene_var > quantile(gene_var, 0.75)]

# ===== 3. 检查样本与基因质量，去除异常值 =====
gsg <- goodSamplesGenes(datExpr1, verbose = 3)
if (!gsg$allOK) {
  datExpr1 <- datExpr1[gsg$goodSamples, gsg$goodGenes]
}
datExpr <- as.data.frame(datExpr1)
nGenes <- ncol(datExpr)
nSamples <- nrow(datExpr)

# ===== 4. 样本聚类，检测离群值（可人工判断是否剔除）=====
trait <- read.csv("96804表型数据.csv", row.names = 1)
trait <- trait[rownames(datExpr), ]  # 保证样本顺序一致

# 假设第一列是样本分组（如 T2D vs Control）
groupList <- factor(trait[, 1])  # 如有多个列可根据你要显示的列修改

# 将组别转为颜色（自动分配颜色）
library(WGCNA)
groupColors <- labels2colors(groupList)

# ===== 5. 样本聚类并绘制彩色分组图 =====
sampleTree <- hclust(dist(datExpr), method = "average")

pdf("Figure1_SampleClustering_ColorGroup.pdf", width = 10, height = 5)
plotDendroAndColors(
  dendro = sampleTree,
  colors = groupColors,
  groupLabels = colnames(trait)[1],
  main = "Sample clustering with trait group color"
)
dev.off()

# ===== 5. soft-threshold (β) 筛选：构建 scale-free 网络结构所需的阈值 =====
powers <- c(1:10, seq(12, 20, by = 2))
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

# 输出每个 β 的 scale-free R²、平均连接度、斜率等指标

softPowerTable <- sft$fitIndices[, c("Power", "SFT.R.sq", "slope", 
                                     "mean.k.", "median.k.", "max.k.")]
colnames(softPowerTable) <- c("Power", "ScaleFree_R2", "Slope", 
                              "MeanConnectivity", "MedianConnectivity", "MaxConnectivity")

# 保存为 CSV
write.csv(softPowerTable, "SoftThreshold_Selection_Table.csv", row.names = FALSE)

# 保存为 CSV 查看
write.csv(softPowerTable, "SoftThreshold_Selection_Table.csv", row.names = FALSE)

# 也在控制台打印，便于查看
print(softPowerTable)

# 可视化两个指标（图与之前一致）
pdf("Figure2_SoftThreshold.pdf", width = 10, height = 5)
par(mfrow = c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3]) * sft$fitIndices[,2],
     xlab="Soft Threshold (power)", ylab="Scale Free Topology Model Fit (R²)", type="n",
     main="Scale Independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3]) * sft$fitIndices[,2],
     labels=powers, cex=0.8, col="red")
abline(h = 0.8, col="blue")

plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)", ylab="Mean Connectivity", type="n",
     main="Mean Connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=0.8, col="red")
dev.off()


# ===== 6. 构建共表达网络，识别模块 =====
# 可根据图中选择的 β 值设置 chosen_power
# 根据经验法则筛选候选 β
candidates <- softPowerTable[
  softPowerTable$ScaleFree_R2 > 0.8 & softPowerTable$MeanConnectivity > 50, 
]

# 若无满足的，则放宽 MeanConnectivity 要求
if (nrow(candidates) == 0) {
  candidates <- softPowerTable[softPowerTable$ScaleFree_R2 > 0.8, ]
}

# 取第一个满足条件的 β 值
chosen_power <- candidates$Power[1]


net <- blockwiseModules(datExpr, power = chosen_power,
                        TOMType = "unsigned", minModuleSize = 30,
                        reassignThreshold = 0, mergeCutHeight = 0.25,
                        numericLabels = TRUE, pamRespectsDendro = FALSE,
                        verbose = 3)

# 转换模块数字为颜色
moduleColors <- labels2colors(net$colors)
table(moduleColors)

# ===== 7. 可视化模块树状图 =====
pdf("Figure3_ModuleDendrogram.pdf", width = 10, height = 5)
plotDendroAndColors(net$dendrograms[[1]], moduleColors[net$blockGenes[[1]]],
                    "Module colors", dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05)
dev.off()

# ===== 8. 保存每个模块内基因列表（供后续葡萄籽打靶、GSEA等） =====
for (color in unique(moduleColors)) {
  write.csv(datExpr[, moduleColors == color],
            paste0("Module_", color, "_genes.csv"), quote = FALSE)
}

# ===== 9. 读入表型数据，并计算模块-性状相关性 =====
# 行名为样本，列为分组/临床指标
trait <- read.csv("96804表型数据.csv", row.names = 1)

# 确保样本顺序一致
trait <- trait[rownames(datExpr), ]

# 计算模块特征基因矩阵
MEs <- moduleEigengenes(datExpr, moduleColors)$eigengenes
MEs <- orderMEs(MEs)

# 计算相关性及p值
modTraitCor <- cor(MEs, trait, use = "p")
modTraitP <- corPvalueStudent(modTraitCor, nSamples)

# 可视化模块-性状热图
textMatrix <- paste0(signif(modTraitCor, 2), "\n(", signif(modTraitP, 1), ")")
dim(textMatrix) <- dim(modTraitCor)

pdf("Figure4_ModuleTraitHeatmap.pdf", width = 9, height = 10)
par(mar = c(5, 10, 4, 2))  # 增大左边距
labeledHeatmap(
  Matrix = modTraitCor,
  xLabels = colnames(trait),
  yLabels = names(MEs),
  ySymbols = names(MEs),
  colorLabels = FALSE,
  colors = blueWhiteRed(50),
  textMatrix = textMatrix,
  setStdMargins = FALSE,
  cex.text = 0.7,
  zlim = c(-1, 1),
  main = "Module-trait relationships"
)
dev.off()

dev.off()
