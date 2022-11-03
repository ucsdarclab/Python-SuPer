import open3d as o3d

dir="results/super/trial_8_face_v2/model29_exp29/trial_8_face_v2_ply/000100.ply"
seman_dir="results/super/trial_8_face_v2/model39_exp39/trial_8_face_v2_seman_ply/000100.ply"

pcd = o3d.io.read_point_cloud(dir)
seman_pcd = o3d.io.read_point_cloud(seman_dir)

