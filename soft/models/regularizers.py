import tensorflow as tf
import utility as ut
import monitor as mn

class Regularizer(object):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.get("alpha",0.001)
        self.regularizers = []
        self.train_zeroing = []

    def _build_reg(self): pass

    def _build_model(self, *args, **kwargs):
        super(Regularizer, self)._build_model(*args, **kwargs)
        for reg in self.regularizers :
            self.cost += reg.alpha*reg._build_reg();
        tf.scalar_summary('Cost',self.cost)

    def depth_l2norms(self, layer):
        for reg in self.regularizers:
            if isinstance(reg, KernelFeatureMap):
                return reg._groups_l2_norm(layer)

        return  0*tf.reduce_sum(tf.square(layer), [0,1]) + 1

    def add_regularizer(self, regularizer):
        regularizer.add_hps(self.hps)
        self.regularizers.append(regularizer)
   
    def add_train_zeroing(self, regularizer):
        regularizer.add_hps(self.hps)
        self.train_zeroing.append(regularizer)
        
    def add_hps(self,hps):
        self.hps = hps

    def zeroing_train(self, tau):
        zeroing_ops = []
        for reg in self.train_zeroing:
            zeroing_ops.append(reg.zeroing(tau))

        return tf.group(*zeroing_ops)

    def zeroing(self, tau):

        zeroing_ops = []
        for reg in self.regularizers:
            zeroing_ops.append(reg.zeroing(tau))

        return tf.group(*zeroing_ops)

    def _groups_l2_norm(self, var, *args, **kwargs):
        l2_norms = 0
        for reg in self.regularizers:
            if isinstance(reg, KernelFeatureMap):
                continue
            if isinstance(reg, RingBasedRegularizer):
                layer_nb = args[0]
                filter_size = args[1]
                l2_norms = reg._groups_l2_norm(var, layer_nb,
                    filter_size)
            elif isinstance(reg, LayerWidth) or isinstance(reg,KernelFeatureMap):
                l2_norms = reg._groups_l2_norm(var)
            elif isinstance(reg, KernelShape):
                l2_norms = reg.group_l2_norm(var,args[0])

        return l2_norms

    def __str__(self):
        verbose ="Regularization :\n"
        for reg in self.regularizers:
            verbose += "-"*5 + str(reg) + "\n"

        return verbose

class GroupBasedRegularizer(Regularizer):
    def _groups_l2_norm(self, layer, layer_nb, group_partitions, flat_dim,
            reduce_dim, filter_size):
        """ Compute L2 norm for nested rings """
        nested_groups =tf.concat(0,[tf.reduce_sum(
            tf.mul(tf.reduce_sum(tf.square(layer),flat_dim),part),
            reduce_dim) for part in group_partitions[layer_nb].get_partition_matrix(filter_size)])
        return tf.sqrt(nested_groups) # [|k_{i1}(G_1)|_2,..., |k_{id_i}(G_[\hat{s_i}]|_2 ]

    def _groups_cardinality(self, nb_kernel, layer_nb, kernel_size, depth, group_partitions):
        return tf.concat(0,[[tf.reduce_sum(part)*depth for part in
                group_partitions[layer_nb].get_partition_matrix(kernel_size)]*nb_kernel])

    def zeroing(self, thresh, group_partitions):
        ops = []
        for var in tf.trainable_variables():
            if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
                # Retrieve Layer parameters
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                nb_kernel = ftype.num_filters
                depth =  ftype.depth
                
                # Retrieve l2 norms of each groupe of each kernels in layer 
                ftype_groups_l2norms = self._groups_l2_norm(var,layer_nb,
                        filter_size)
            
                
                # Generate the zeroing matrix relative to the computed kernel # size
                indicator =tf.cast(tf.greater_equal(ftype_groups_l2norms,thresh),tf.float32)
                groups_indicator = tf.add_n(tf.split(0,filter_size,indicator))
                zeroing_matrix_ind =[tf.cast(
                                        tf.equal(groups_indicator,i),
                                        tf.float32) 
                                        for i in range(filter_size + 1)]

                zeroing_matrix = ut.generate_groups_zeroing_matrix(zeroing_matrix_ind, 
                        filter_size, depth, nb_kernel, layer_nb,
                        group_partitions)
                ops.append(var.assign(tf.mul(var,zeroing_matrix)))
                
        return tf.group(*ops)


class RingBasedRegularizer(GroupBasedRegularizer):
    def _groups_l2_norm(self, layer, layer_nb, filter_size):
        return super(RingBasedRegularizer, self)._groups_l2_norm(layer,
                layer_nb, ut.ring_partitions, 2, [0,1], filter_size)

    def zeroing(self, thresh):
        return super(RingBasedRegularizer, self).zeroing(thresh, ut.ring_partitions)

    def _groups_cardinality(self, nb_kernel, layer_nb, kernel_size, depth):
        return super(RingBasedRegularizer, self)._groups_cardinality(nb_kernel,
                layer_nb, kernel_size, depth, ut.ring_partitions)
    
class WeightDecay(Regularizer):
    def _build_reg(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0 and var.op.name.find('conv') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.add_n(costs)

    def __str__(self):
        return "weight decay"

class Lasso(Regularizer):
    def _build_reg(self):
        """L1 penalization"""
        costs = []
        for var in tf.trainable_variables():
            if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
                costs.append(tf.reduce_sum(tf.abs(var)))
        return tf.add_n(costs)

    def __str__(self):
        return "lasso"

class LayerWidth(Regularizer):

    def _groups_l2_norm(self, layer):
        return tf.sqrt(tf.reduce_sum(tf.square(layer), [0,1,2])) 

    def _build_reg(self):
        """L2,1 loss on the number of kernels. Should enforce their sparsity"""
        
        kernel_l2_norms=[]
        for var in tf.trainable_variables():
            if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                
                layer_kernel_l2_norms = tf.sqrt(tf.reduce_sum(tf.square(var), [0,1,2])) # [ |k_{i1}|_2, ..., |k_{id_i}|_2 ]
                tf.histogram_summary('Kernel l2 norms - '+var.op.name, layer_kernel_l2_norms) ### CB2:SPEED
                mn.count_kernel_tau(layer_kernel_l2_norms, layer_nb)

                kernel_l2_norms.append(tf.reduce_sum(layer_kernel_l2_norms))
        
        return tf.add_n(kernel_l2_norms)
    
    def zeroing(self, thresh):
        """Kernel clamping to 0 when considered as empty in term of l2 norm"""

        ops = []
        for var in tf.trainable_variables():
            if (var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)

                individual_l2loss =  self._groups_l2_norm(var)
                indicator = tf.cast(tf.greater_equal(individual_l2loss, thresh), tf.float32)

                zeroing_tensor =ut.generate_zeroing_matrix(self.expand(indicator),
                                                            filter_size,
                                                            filter_size,
                                                            ftype.depth)

                ops.append(var.assign(tf.mul(var, zeroing_tensor)))
                
        return tf.group(*ops)

    def expand(self, ind):
        for _ in range(3):
            ind = tf.expand_dims(ind, 0)

        return ind

    def __str__(self):
        return "group lasso : layer width"
        

class KernelSize(RingBasedRegularizer):
    def _build_reg(self):
        """Kernel size base regularization"""
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                layer_groups_l2norm = self._groups_l2_norm(var,
                    layer_nb, filter_size)
                mn.kernel_size_distribution(layer_groups_l2norm, filter_size,
                        layer_nb, ftype.num_filters)
                l2_norms.append(tf.reduce_sum(layer_groups_l2norm))

        tf.histogram_summary('l2_norms',l2_norms)
        return tf.add_n(l2_norms)

    def __str__(self):
        return "group lasso : kernel size"

class KernelSizeEnhanced(RingBasedRegularizer):
    def _build_reg(self):
        """Kernel size base regularization with l_1/2 norm"""
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                layer_groups_l2norm = self._groups_l2_norm(var,
                    layer_nb, filter_size)
                mn.kernel_size_distribution(layer_groups_l2norm, filter_size,
                        layer_nb, ftype.num_filters)
                l2_norms.append(tf.reduce_sum(tf.sqrt(layer_groups_l2norm)))

        tf.histogram_summary('l2_norms',l2_norms)
        return tf.add_n(l2_norms)

    def __str__(self):
        return "group l_1/2 : kernel size"


class KernelSizeLSP(RingBasedRegularizer):
    def _build_reg(self):
        """Kernel size base regularization"""
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                layer_groups_l2norm = self._groups_l2_norm(var,
                    layer_nb, filter_size)
                mn.kernel_size_distribution(layer_groups_l2norm, filter_size,
                        layer_nb, ftype.num_filters)
                l2_norms.append(tf.reduce_sum(tf.log(layer_groups_l2norm)))

        tf.histogram_summary('l2_norms',l2_norms)
        return tf.add_n(l2_norms)

    def __str__(self):
        return "group LSP : kernel size"

class KernelSizeFastLearning(RingBasedRegularizer):
    def _build_reg(self):
        """Kernel size base regularization"""
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                layer_groups_l2norm = self._groups_l2_norm(var,
                    layer_nb, filter_size)
                mn.kernel_size_distribution(layer_groups_l2norm, filter_size,
                        layer_nb, ftype.num_filters)
                l2_norms.append(tf.reduce_sum(tf.log(layer_groups_l2norm)))

        tf.histogram_summary('l2_norms',l2_norms)
        return tf.add_n(l2_norms)

    def __str__(self):
        return "group LSP : kernel size"

class WeightedKernelSize(RingBasedRegularizer):
    def _build_reg(self):
        """Kernel size base regularization"""

        layer_nb = 0
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)

                rings_l2norms = self._groups_l2_norm(var,layer_nb, filter_size)
                mn.kernel_size_distribution(rings_l2norms, filter_size,
                        layer_nb, ftype.num_filters)
                groups_cardinality = self._groups_cardinality(ftype.num_filters,
                        layer_nb, filter_size, ftype.depth)
                l2_norms.append(tf.reduce_sum(tf.mul(rings_l2norms,
                    tf.sqrt(groups_cardinality))))
                

        return tf.add_n(l2_norms)


    def __str__(self):
        return "group lasso : weighted kernel size"

class NormalizedKernelSize(RingBasedRegularizer):
    def _build_reg(self):
        """Kernel size base regularization"""

        layer_nb = 0
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                
                rings_l2norms = NormalizedKernelSize._groups_l2_norm(self,
                        var, layer_nb, filter_size)
                mn.kernel_size_distribution(rings_l2norms, filter_size,
                        layer_nb, ftype.num_filters)
                groups_cardinality =NormalizedKernelSize._groups_cardinality(self,
                        ftype.num_filters, layer_nb, filter_size, ftype.depth)
                l2_norms.append(tf.reduce_sum(tf.div(rings_l2norms,tf.sqrt(groups_cardinality))))
                

        return tf.add_n(l2_norms)

    def __str__(self):
        return "group lasso : normalized kernel size"

class Complexity(RingBasedRegularizer):
    def _build_reg(self):
        """Complexity based L2,1 regularization"""

        l2_norms = []
        nb_kernel_prev = 1.0
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                layer = self.hps.get_layer(layer_nb, filter_size)
                in_fmap_size = self.hps.layers[layer_nb].fmap

                rings_l2norms = Complexity._groups_l2_norm(self, var,layer_nb,
                    filter_size)
                layer_groups_l2norm = tf.add_n(tf.split(0,layer.filter_size,rings_l2norms))
               
                nb_kernel = tf.reduce_sum(ut.kernel_l2_loss(var))

                l2_norms.append(tf.mul(in_fmap_size**2,
                    tf.mul(nb_kernel_prev,tf.reduce_sum(tf.square(layer_groups_l2norm)))
                    ))
                
                nb_kernel_prev = nb_kernel

        return tf.add_n(l2_norms)

class ComplexityBis(RingBasedRegularizer):
    def _build_reg(self):
        """Complexity based L2,1 regularization"""

        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                layer = self.hps.get_layer(layer_nb, filter_size)
                in_fmap_size = self.hps.layers[layer_nb].fmap

                rings_l2norms = ComplexityBis._groups_l2_norm(self, var,layer_nb,
                    filter_size)
                layer_groups_l2norm = tf.add_n(tf.split(0,layer.filter_size,rings_l2norms))
               

                l2_norms.append(tf.mul(1.0/(in_fmap_size**2),
                    tf.reduce_sum(tf.square(layer_groups_l2norm)))
                    )
                

        return tf.add_n(l2_norms)



class KernelShape(GroupBasedRegularizer):
    def __init__(self,*args, **kwargs):
        super(KernelShape, self).__init__(*args, **kwargs)
        self.turn = 'col'

    def _groups_l2_norm(self, var, layer_nb, filter_size):
        if self.turn == 'col':
            return super(KernelShape, self)._groups_l2_norm(var, layer_nb,
                        ut.shape_partitions, [2,1], 0, filter_size)
        else:
            return super(KernelShape, self)._groups_l2_norm(var, layer_nb,
                        ut.shape_partitions, [2,0], 0, filter_size)


    def group_l2_norm(self, layer, direction):
        self.turn = direction
        layer_nb, filter_size = ut.extract_filter_prop(layer.op.name)
        return self._groups_l2_norm(layer, layer_nb, filter_size)

    def _build_reg(self):
        l2_norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                ftype = self.hps.get_layer(layer_nb, filter_size)
                nb_kernel = ftype.num_filters
                depth = ftype.depth

                #direction_card = self._groups_cardinality(layer_nb,
                #        filter_size, depth, nb_kernel)
                col_l2norms = self._groups_l2_norm(var, layer_nb, filter_size)
                self.turn = 'row'
                row_l2norms = self._groups_l2_norm(var, layer_nb, filter_size)
                self.turn = 'col'
                l2_norms.append(
                        tf.reduce_sum(col_l2norms) +
                        tf.reduce_sum(row_l2norms)
                        )

        return tf.add_n(l2_norms)

    def _groups_cardinality(self, layer_nb, filter_size, depth, nb_kernel) :
        return tf.mul(float(filter_size), super(KernelShape, self).
                _groups_cardinality(nb_kernel, layer_nb, filter_size, depth, ut.shape_partitions))

    def zeroing(self, threshold):
        col_row_ops = []
        partitions_expanded = self.expand()
        col_row_ops.append(super(KernelShape, self).zeroing(threshold,
                partitions_expanded))
        self.turn = 'row'
        col_row_ops.append(super(KernelShape, self).zeroing(threshold,
                partitions_expanded))
        self.turn = 'col'

        return tf.group(*col_row_ops)

    def __str__(self):
        return "group lasso : kernel shape"
    
    def expand(self):
        """
        Expand dimension of partition matrix to fit utility function dimension
        requirements
        """
        partitions = []
        for layer in ut.shape_partitions:
            partitions.append(layer.expand(2))

        return partitions

class KernelFeatureMap(Regularizer):

    def _groups_l2_norm(self, var):
        return tf.sqrt(tf.reduce_sum(tf.square(var), [0,1]))

    def _build_reg(self):
        l2norms = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                l2norms.append(tf.reduce_sum(KernelFeatureMap._groups_l2_norm(self,var)))
        return tf.add_n(l2norms)

    def zeroing(self, tau):
        zeroing_ops = []
        for var in tf.trainable_variables():
            if ((var.op.name.find('DW') > 0) and (var.op.name.find('conv') > 0)):
                layer_nb, filter_size = ut.extract_filter_prop(var.op.name)
                layer = self.hps.get_layer(layer_nb, filter_size)
                groups_l2norms = KernelFeatureMap._groups_l2_norm(self,var)
                groups_indicator = tf.cast(tf.greater_equal(groups_l2norms,
                    tau), tf.float32)
                zeroing_matrix = ut.n_dim_zeroing(2,groups_indicator,layer.filter_size,
                        layer.filter_size, 0)
                zeroing_ops.append(tf.mul(zeroing_matrix, var))
                

        return tf.group(*zeroing_ops)
    
    def __str__(self):
        return "group lasso : kernel feature map"
