# Physics informed neural networks for extreme mechanics problems
Jeongsu Lee

Department of mechanical, smart, and industrial engineering, Gachon University, Seongnam 13120, South Korea
Corresponding authors: leejeongsu@gachon.ac.kr 

Abstract
Physics-informed neural networks (PINNs) constitute a continuous and differentiable mapping function that approximates a solution curve to given physical laws. Although recent extensive studies have exhibited great potential for PINNs as an alternative or complement to conventional numerical methods, the lack of accuracy and robustness remains a challenging problem. As a remedy, this study explores optimal strategies for constructing PINNs by resolving several extreme problems where the solution based on a regular PINN was not very effective. Examples include dynamic beam bending, fluid-structure interaction, and 1-D advection in high velocity. The novel strategies proposed are: 1) scaling analysis for the configuration, 2) incorporation of the second-order non-linear term of input variables, and 3) use of a neural network architecture that reflects a series solution of decomposed bases. The proposed approaches are shown to significantly improve PINN predictions, exhibiting an order-of-magnitude improvement in accuracy compared to regular PINNs. This study is expected to provide crucial baselines for constructing PINNs, which could be extended to high-order coupled differential equations.
