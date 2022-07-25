from super.loss import *

# from RAFT.core.utils.flow_viz import flow_to_image

from utils.config import *
from utils.utils import *


# def corr(fmap1, fmap2, fdim=1):
#     return torch.matmul(fmap1, fmap2.permute(-1,-2))

def graph_fit(model_args, src, trg, Niter=10, nets=None, print_loss=False, trg_=None):
    method = model_args['method']

    src_graph = src.ED_nodes
    src_edge_index = src_graph.edge_index.type(long_)

    sf_knn_indices = src.knn_indices[src.isStable]
    sf_knn_w = src.knn_w[src.isStable]
    sf_knn = src_graph.points[sf_knn_indices]
    sf_diff = src.points[src.isStable].unsqueeze(1) - sf_knn
    skew_v = get_skew(sf_diff)

    # if model_args['sf-corr'][0]:
    #     if "optical_flow" in nets:
    #         optical_flow_net = nets["optical_flow"]
    #         flow = optical_flow_net(
    #             de_normalize(sfModel.renderImg), de_normalize(trg_data.rgb))[1]
    #         flow = flow[0].permute(1,2,0)
    #     else:
    #         flow=None
    
    # if method == 'dlsuper' and False:
    #     # TODO: better way to get fcorr.
    #     fcorr = corr(
    #         F.normalize(src.x)[0].permute(1,2,0)[src.valid],
    #         F.normalize(trg.x)[0].permute(1,2,0)[trg.valid]
    #         ) # n x m
    #     fcorr = fcorr**2

    """
    Optimization loop.
    """
    deform_verts = torch.tensor(
        np.repeat(np.array([[1.,0.,0.,0.,0.,0.,0.]]), src_graph.num, axis=0),
        dtype=fl64_, device=dev, requires_grad=True)
    # Init optimizer.
    optimizer = torch.optim.SGD([deform_verts], lr=0.00005, momentum=0.9)
        
    for i in range(Niter):
        optimizer.zero_grad()
        
        # Deform the mesh and surfels.
        new_verts = src_graph.points + deform_verts[:,4:]
        new_norms, _ = transformQuatT(src_graph.norms, deform_verts[...,0:4])
        
        new_sf, _ = Trans_points(sf_diff, sf_knn, deform_verts[sf_knn_indices], sf_knn_w, skew_v=skew_v)
        new_sfnorms = torch.sum(new_norms[sf_knn_indices] * sf_knn_w.unsqueeze(-1), dim=1)
        
        new_data = Data(points=new_sf, norms=new_sfnorms)
        # if model_args['method'] == 'seman-super':
        #     new_data.seg = src.seg

        """
        Mesh losses.
        """
        loss_mesh = 0.0
        
        # # Point-point loss.
        # if model_args['m-point-point'][0]:
        #     radius = 1.0
        #     src2trg_mesh_dists, _ = find_knn(new_verts, trg.points, k=5)
        #     weights = torch.exp(-src2trg_mesh_dists.detach()/(radius**2))
        #     loss_mesh += model_args['m-point-point'][1] * (weights * src2trg_mesh_dists).sum()

        # # Point-plane loss.
        # if model_args['m-point-plane'][0]:
        #     radius = 1.0
        #     src2trg_mesh_dists, src2trg_mesh_idx = find_knn(new_verts, trg.points, k=5)
        #     weights = torch.exp(-src2trg_mesh_dists.detach()/(radius**2))
        #     loss_mesh += model_args['m-point-plane'][1] * (weights * torch_inner_prod(
        #         trg.norms[src2trg_mesh_idx],
        #         new_verts.unsqueeze(1) - trg.points[src2trg_mesh_idx]
        #     )**2).sum()

        # # Edge loss.
        # if model_args['m-edge'][0]:
        #     loss_mesh += model_args['m-edge'][1] * ((torch_distance(
        #         new_verts[src_edge_index[:,0]], new_verts[src_edge_index[:,1]])
        #         - src.edges_lens) ** 2).sum()

        # # TODO Face loss. (Might not be very useful though.)

        # if model_args['m-point-point'][0]:
        #     loss_mesh = model_args['m-point-point'][1] * (fcorr * torch_sq_distance(
        #         new_verts.unsqueeze(1), trg.points.unsqueeze(0))
        #         ).sum() # Point-point loss.

        # if model_args['m-point-plane'][0]:
        #     loss_mesh = model_args['m-point-plane'][1] * (fcorr * torch_inner_prod(
        #         trg.norms.unsqueeze(0), 
        #         new_verts.unsqueeze(1) - trg.points.unsqueeze(0)
        #         )**2).sum() # Point-plane loss.

        # Regularization terms.
        if model_args['m-arap'][0]:
            loss_mesh += model_args['m-arap'][1] * \
                ARAPLoss.autograd_forward(src_graph, deform_verts)

        if model_args['m-rot'][0]:
            loss_mesh += model_args['m-rot'][1] * RotLoss.autograd_forward(deform_verts).sum()

        """
        Surfel losses.
        """
        loss_surfels = 0.0

        # Surfel point-plane loss.
        if model_args['sf-point-plane'][0]:
            loss_surfels += model_args['sf-point-plane'][1] * \
                DataLoss.autograd_forward(model_args, new_data, trg)

        # Correlation loss. (Optical flow-based)
        if model_args['sf-corr'][0]:
            loss_surfels += model_args['sf-corr'][1] * \
                DataLoss.autograd_forward(new_data, trg, flow=flow, loss_type='point-point')
        
        # TODO: Try mesh normal consistency loss and mesh laplacian smoothing loss.
        # loss_normal = mesh_normal_consistency(new_src_mesh)
        # loss_laplacian = mesh_laplacian_smoothing(full_new_src_mesh, method="uniform")

        # Weighted sum of the losses
        loss = loss_mesh + loss_surfels
        if print_loss:
            print("     Mesh: {:.02f}, Surfel: {:.02f}".format(loss_mesh, loss_surfels))
            
        # Optimization step.
        loss.backward(retain_graph=True)
        optimizer.step()
        torch.cuda.empty_cache()

    return deform_verts