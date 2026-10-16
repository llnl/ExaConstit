#include "utilities/mechanics_kernels.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "RAJA/RAJA.hpp"

#include <string>
#include <gtest/gtest.h>

using namespace std;
using namespace mfem;

static int outputLevel = 0;

//Routine to test the deformation gradient to make sure we're getting out the right values.
//This applies the following displacement field to the nodal values:
//u_vec = (2x + 3y + 4z)i + (4x + 2y + 3z)j + (3x + 4y + 2z)k
void test_deformation_field_set(ParGridFunction *gf, ParGridFunction *disp)
{
   
   const double* temp_vals = gf->HostRead();
   double* vals = disp->HostReadWrite();

   int dim = gf->Size()/3;

   for (int i = 0; i < dim; ++i)
   {
      const double x1 = temp_vals[i];
      const double x2 = temp_vals[i + dim];
      const double x3 = temp_vals[i + 2 * dim];
      
      vals[i] = 2 * x1 + 3 * x2 + 4 * x3;
      vals[i + dim] = 4 * x1 + 2 * x2 + 3 * x3;
      vals[i + 2 * dim] = 3 * x1 + 4 * x2 + 2 * x3;
   }
}

// This function had to be moved out of the TEST() macro
// as CUDA was now complains about it being a private function/variable.
// Therefore, we couldn't have our MFEM_FORALL loops in there.
double test_main_body()
{
   int dim = 3;
   int order = 3;
   mfem::ParMesh *pmesh = nullptr;
   {
      // Making this mesh and test real simple with 8 cubic element
      mfem::Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
      mesh.SetCurvature(order);
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, mesh);
   }

   mfem::H1_FECollection fec(order, dim);
   mfem::ParFiniteElementSpace fes(pmesh, &fec, dim);

   mfem::ParGridFunction x_ref(&fes);
   mfem::ParGridFunction x_cur(&fes);
   mfem::ParGridFunction disp(&fes);

   pmesh->GetNodes(x_ref);

   test_deformation_field_set(&x_ref, &disp);

   x_cur = x_ref;
   x_cur += disp;

   {
      // fix me: should the mesh nodes be on the device?
      GridFunction *nodes = &x_cur; // set a nodes grid function to global current configuration
      int owns_nodes = 0;
      pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
   }

   const int intOrder = 2 * order + 1;
   std::shared_ptr<mfem::QuadratureSpace> qspace = std::make_shared<mfem::QuadratureSpace>(mfem::ptr_utils::borrow_ptr(pmesh), intOrder);
   mfem::QuadratureFunction raderiv(qspace, 9);
   mfem::QuadratureFunction rderiv(qspace, 9);

   const IntegrationRule *ir = &(IntRules.Get(fes.GetTypicalFE()->GetGeomType(), intOrder));

   const int nqpts = ir->GetNPoints();
   const int ndofs = fes.GetTypicalFE()->GetDof();
   const int nelems = fes.GetNE();
   const int space_dims = fes.GetTypicalFE()->GetDim();

   {
      auto coord = mfem::Reshape(raderiv.ReadWrite(), 3, 3, nqpts, nelems);
      // u_vec = (2x + 3y + 4z)i + (4x + 2y + 3z)j + (3x + 4y + 2z)k
      mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int i) {
         for(int j = 0; j < nqpts; j++) {
            coord(0, 0, j, i) = 3.0;
            coord(0, 1, j, i) = 3.0;
            coord(0, 2, j, i) = 4.0;

            coord(1, 0, j, i) = 4.0;
            coord(1, 1, j, i) = 3.0;
            coord(1, 2, j, i) = 3.0;

            coord(2, 0, j, i) = 3.0;
            coord(2, 1, j, i) = 4.0;
            coord(2, 2, j, i) = 3.0;
         }
      });
   }

   {
      // fix me: should the mesh nodes be on the device?
      mfem::GridFunction *nodes = &x_ref; // set a nodes grid function to global current configuration
      int owns_nodes = 0;
      pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
   }

   {
      // We need to make sure these are deleted at the start of each iteration
      // since we have meshes that are constantly changing.
      pmesh->DeleteGeometricFactors();
      const mfem::GeometricFactors *geom = pmesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);
      mfem::Vector xcur(fes.TrueVSize());
      x_cur.GetTrueDofs(xcur);

      // So, we're going to originally support non tensor-product type elements originally.
      const mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE;
      // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const mfem::Operator *elem_restrict_lex; // Not owned
      const mfem::Operator *P;

      elem_restrict_lex = fes.GetElementRestriction(ordering);
      P = fes.GetProlongationMatrix();

      mfem::Vector el_x;
      el_x.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      mfem::Vector px, el_jac;
      el_x.UseDevice(true);
      px.SetSize(P->Height(), Device::GetMemoryType());
      px.UseDevice(true);
      el_jac.SetSize(space_dims * space_dims * nqpts * nelems, Device::GetMemoryType());
      el_jac.UseDevice(true);

      // Takes in k vector and transforms into into our E-vector array
      P->Mult(xcur, px);
      elem_restrict_lex->Mult(px, el_x);

      // geom->J really isn't going to work for us as of right now. We could just reorder it
      // to the version that we want it to be in instead...

      const int DIM4 = 4;
      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ space_dims, space_dims, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > jac_view(el_jac.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, space_dims, space_dims, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

      MFEM_FORALL(i, nelems,
      {
         const int nqpts_ = nqpts;
         const int space_dims_ = space_dims;
         for (int j = 0; j < nqpts_; j++) {
            for (int k = 0; k < space_dims_; k++) {
               for (int l = 0; l < space_dims_; l++) {
                  jac_view(l, k, j, i) = geom_j_view(j, l, k, i);
               }
            }
         }
      });

      const FiniteElement &el = *fes.GetTypicalFE();
      Vector qpts_dshape;
      qpts_dshape.SetSize(nqpts * space_dims * ndofs, Device::GetMemoryType());
      qpts_dshape.UseDevice(true);
      {
         DenseMatrix DSh;
         const int offset = ndofs * space_dims;
         double *qpts_dshape_data = qpts_dshape.HostReadWrite();
         for (int i = 0; i < nqpts; i++) {
            const IntegrationPoint &ip = ir->IntPoint(i);
            DSh.UseExternalData(&qpts_dshape_data[offset * i], ndofs, space_dims);
            el.CalcDShape(ip, DSh);
         }
      }
      rderiv = 0.0;
      exaconstit::kernel::GradCalc(nqpts, nelems, ndofs, el_jac.Read(), qpts_dshape.Read(), el_x.Read(), rderiv.ReadWrite());
   }

   raderiv -= rderiv;

   const double difference = raderiv.Norml2() / raderiv.Size();

   return difference;
}

TEST(exaconstit, gradient)
{
   const double difference = test_main_body();
   EXPECT_LT(fabs(difference), 3e-15) << "Did not get expected value for pa vec";
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // Testing the case for a dense CMat and then a sparser version of CMat

   Device device("cpu");
   printf("\n");
   device.Print();

   ::testing::InitGoogleTest(&argc, argv);
   if (argc > 1) {
      outputLevel = atoi(argv[1]);
   }
   std::cout << "got outputLevel : " << outputLevel << std::endl;

   int i = RUN_ALL_TESTS();

   MPI_Finalize();

   return i;
}
