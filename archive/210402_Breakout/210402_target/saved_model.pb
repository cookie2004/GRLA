¤
¿£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8Â

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
: *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:@*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:@*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
À*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0

NoOpNoOp
­
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*è
valueÞBÛ BÔ
Î
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
 
F
0
1
2
3
4
5
$6
%7
*8
+9
F
0
1
2
3
4
5
$6
%7
*8
+9
 
­

0layers
1layer_regularization_losses
		variables
2layer_metrics

trainable_variables
regularization_losses
3non_trainable_variables
4metrics
 
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

5layers
6layer_regularization_losses
	variables
7layer_metrics
trainable_variables
regularization_losses
8non_trainable_variables
9metrics
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

:layers
;layer_regularization_losses
	variables
<layer_metrics
trainable_variables
regularization_losses
=non_trainable_variables
>metrics
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

?layers
@layer_regularization_losses
	variables
Alayer_metrics
trainable_variables
regularization_losses
Bnon_trainable_variables
Cmetrics
 
 
 
­

Dlayers
Elayer_regularization_losses
 	variables
Flayer_metrics
!trainable_variables
"regularization_losses
Gnon_trainable_variables
Hmetrics
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
­

Ilayers
Jlayer_regularization_losses
&	variables
Klayer_metrics
'trainable_variables
(regularization_losses
Lnon_trainable_variables
Mmetrics
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
­

Nlayers
Olayer_regularization_losses
,	variables
Player_metrics
-trainable_variables
.regularization_losses
Qnon_trainable_variables
Rmetrics
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_4Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿTT
è
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_48119748
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_48120041
Â
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_48120081á

þ
J__inference_functional_7_layer_call_and_return_conditional_losses_48119643

inputs
conv2d_9_48119616
conv2d_9_48119618
conv2d_10_48119621
conv2d_10_48119623
conv2d_11_48119626
conv2d_11_48119628
dense_6_48119632
dense_6_48119634
dense_7_48119637
dense_7_48119639
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_48119616conv2d_9_48119618*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_9_layer_call_and_return_conditional_losses_481194422"
 conv2d_9/StatefulPartitionedCallÊ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_48119621conv2d_10_48119623*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_481194692#
!conv2d_10/StatefulPartitionedCallË
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_48119626conv2d_11_48119628*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_481194962#
!conv2d_11/StatefulPartitionedCallþ
flatten_3/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_481195182
flatten_3/PartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_48119632dense_6_48119634*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_481195372!
dense_6/StatefulPartitionedCall·
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_48119637dense_7_48119639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_481195632!
dense_7/StatefulPartitionedCall«
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
³
­
E__inference_dense_6_layer_call_and_return_conditional_losses_48119537

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ù
ü
/__inference_functional_7_layer_call_fn_48119878

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_7_layer_call_and_return_conditional_losses_481196982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
¿
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_48119944

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ
­
E__inference_dense_7_layer_call_and_return_conditional_losses_48119563

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
®
F__inference_conv2d_9_layer_call_and_return_conditional_losses_48119442

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿTT:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
Ñ
­
E__inference_dense_7_layer_call_and_return_conditional_losses_48119979

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
ý
/__inference_functional_7_layer_call_fn_48119666
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_7_layer_call_and_return_conditional_losses_481196432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_4
	
¯
G__inference_conv2d_10_layer_call_and_return_conditional_losses_48119909

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶"
Ä
!__inference__traced_save_48120041
file_prefix.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_660c837dc9b2458284fca45a89c75e2d/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÇ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ù
valueÏBÌB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesò
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesr
p: : : : @:@:@@:@:
À::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
À:!

_output_shapes	
::%	!

_output_shapes
:	: 


_output_shapes
::

_output_shapes
: 
³
­
E__inference_dense_6_layer_call_and_return_conditional_losses_48119960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¿-

$__inference__traced_restore_48120081
file_prefix$
 assignvariableop_conv2d_9_kernel$
 assignvariableop_1_conv2d_9_bias'
#assignvariableop_2_conv2d_10_kernel%
!assignvariableop_3_conv2d_10_bias'
#assignvariableop_4_conv2d_11_kernel%
!assignvariableop_5_conv2d_11_bias%
!assignvariableop_6_dense_6_kernel#
assignvariableop_7_dense_6_bias%
!assignvariableop_8_dense_7_kernel#
assignvariableop_9_dense_7_bias
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Í
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ù
valueÏBÌB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_11_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_11_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_7_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_7_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	
®
F__inference_conv2d_9_layer_call_and_return_conditional_losses_48119889

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿTT:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
	
¯
G__inference_conv2d_11_layer_call_and_return_conditional_losses_48119496

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ		@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs

ÿ
J__inference_functional_7_layer_call_and_return_conditional_losses_48119610
input_4
conv2d_9_48119583
conv2d_9_48119585
conv2d_10_48119588
conv2d_10_48119590
conv2d_11_48119593
conv2d_11_48119595
dense_6_48119599
dense_6_48119601
dense_7_48119604
dense_7_48119606
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall£
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_9_48119583conv2d_9_48119585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_9_layer_call_and_return_conditional_losses_481194422"
 conv2d_9/StatefulPartitionedCallÊ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_48119588conv2d_10_48119590*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_481194692#
!conv2d_10/StatefulPartitionedCallË
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_48119593conv2d_11_48119595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_481194962#
!conv2d_11/StatefulPartitionedCallþ
flatten_3/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_481195182
flatten_3/PartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_48119599dense_6_48119601*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_481195372!
dense_6/StatefulPartitionedCall·
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_48119604dense_7_48119606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_481195632!
dense_7/StatefulPartitionedCall«
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_4
ù
ü
/__inference_functional_7_layer_call_fn_48119853

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_7_layer_call_and_return_conditional_losses_481196432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs


+__inference_conv2d_9_layer_call_fn_48119898

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_9_layer_call_and_return_conditional_losses_481194422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿTT::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
	
¯
G__inference_conv2d_10_layer_call_and_return_conditional_losses_48119469

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª
H
,__inference_flatten_3_layer_call_fn_48119949

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_481195182
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ÿ
J__inference_functional_7_layer_call_and_return_conditional_losses_48119580
input_4
conv2d_9_48119453
conv2d_9_48119455
conv2d_10_48119480
conv2d_10_48119482
conv2d_11_48119507
conv2d_11_48119509
dense_6_48119548
dense_6_48119550
dense_7_48119574
dense_7_48119576
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall£
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_9_48119453conv2d_9_48119455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_9_layer_call_and_return_conditional_losses_481194422"
 conv2d_9/StatefulPartitionedCallÊ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_48119480conv2d_10_48119482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_481194692#
!conv2d_10/StatefulPartitionedCallË
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_48119507conv2d_11_48119509*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_481194962#
!conv2d_11/StatefulPartitionedCallþ
flatten_3/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_481195182
flatten_3/PartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_48119548dense_6_48119550*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_481195372!
dense_6/StatefulPartitionedCall·
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_48119574dense_7_48119576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_481195632!
dense_7/StatefulPartitionedCall«
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_4


,__inference_conv2d_11_layer_call_fn_48119938

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_481194962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ		@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
à

*__inference_dense_7_layer_call_fn_48119988

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_481195632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â

*__inference_dense_6_layer_call_fn_48119969

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_481195372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

þ
J__inference_functional_7_layer_call_and_return_conditional_losses_48119698

inputs
conv2d_9_48119671
conv2d_9_48119673
conv2d_10_48119676
conv2d_10_48119678
conv2d_11_48119681
conv2d_11_48119683
dense_6_48119687
dense_6_48119689
dense_7_48119692
dense_7_48119694
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_48119671conv2d_9_48119673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_9_layer_call_and_return_conditional_losses_481194422"
 conv2d_9/StatefulPartitionedCallÊ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_48119676conv2d_10_48119678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_481194692#
!conv2d_10/StatefulPartitionedCallË
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_48119681conv2d_11_48119683*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_481194962#
!conv2d_11/StatefulPartitionedCallþ
flatten_3/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_481195182
flatten_3/PartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_48119687dense_6_48119689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_481195372!
dense_6/StatefulPartitionedCall·
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_48119692dense_7_48119694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_481195632!
dense_7/StatefulPartitionedCall«
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
Ì
ô
&__inference_signature_wrapper_48119748
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_481194272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_4
¯2

#__inference__wrapped_model_48119427
input_48
4functional_7_conv2d_9_conv2d_readvariableop_resource9
5functional_7_conv2d_9_biasadd_readvariableop_resource9
5functional_7_conv2d_10_conv2d_readvariableop_resource:
6functional_7_conv2d_10_biasadd_readvariableop_resource9
5functional_7_conv2d_11_conv2d_readvariableop_resource:
6functional_7_conv2d_11_biasadd_readvariableop_resource7
3functional_7_dense_6_matmul_readvariableop_resource8
4functional_7_dense_6_biasadd_readvariableop_resource7
3functional_7_dense_7_matmul_readvariableop_resource8
4functional_7_dense_7_biasadd_readvariableop_resource
identity×
+functional_7/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4functional_7_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+functional_7/conv2d_9/Conv2D/ReadVariableOpç
functional_7/conv2d_9/Conv2DConv2Dinput_43functional_7/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
functional_7/conv2d_9/Conv2DÎ
,functional_7/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5functional_7_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,functional_7/conv2d_9/BiasAdd/ReadVariableOpà
functional_7/conv2d_9/BiasAddBiasAdd%functional_7/conv2d_9/Conv2D:output:04functional_7/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_7/conv2d_9/BiasAdd¢
functional_7/conv2d_9/ReluRelu&functional_7/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_7/conv2d_9/ReluÚ
,functional_7/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5functional_7_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,functional_7/conv2d_10/Conv2D/ReadVariableOp
functional_7/conv2d_10/Conv2DConv2D(functional_7/conv2d_9/Relu:activations:04functional_7/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
2
functional_7/conv2d_10/Conv2DÑ
-functional_7/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6functional_7_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-functional_7/conv2d_10/BiasAdd/ReadVariableOpä
functional_7/conv2d_10/BiasAddBiasAdd&functional_7/conv2d_10/Conv2D:output:05functional_7/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2 
functional_7/conv2d_10/BiasAdd¥
functional_7/conv2d_10/ReluRelu'functional_7/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
functional_7/conv2d_10/ReluÚ
,functional_7/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5functional_7_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,functional_7/conv2d_11/Conv2D/ReadVariableOp
functional_7/conv2d_11/Conv2DConv2D)functional_7/conv2d_10/Relu:activations:04functional_7/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
functional_7/conv2d_11/Conv2DÑ
-functional_7/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6functional_7_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-functional_7/conv2d_11/BiasAdd/ReadVariableOpä
functional_7/conv2d_11/BiasAddBiasAdd&functional_7/conv2d_11/Conv2D:output:05functional_7/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
functional_7/conv2d_11/BiasAdd¥
functional_7/conv2d_11/ReluRelu'functional_7/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_7/conv2d_11/Relu
functional_7/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
functional_7/flatten_3/ConstÐ
functional_7/flatten_3/ReshapeReshape)functional_7/conv2d_11/Relu:activations:0%functional_7/flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2 
functional_7/flatten_3/ReshapeÎ
*functional_7/dense_6/MatMul/ReadVariableOpReadVariableOp3functional_7_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02,
*functional_7/dense_6/MatMul/ReadVariableOpÔ
functional_7/dense_6/MatMulMatMul'functional_7/flatten_3/Reshape:output:02functional_7/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_7/dense_6/MatMulÌ
+functional_7/dense_6/BiasAdd/ReadVariableOpReadVariableOp4functional_7_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_7/dense_6/BiasAdd/ReadVariableOpÖ
functional_7/dense_6/BiasAddBiasAdd%functional_7/dense_6/MatMul:product:03functional_7/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_7/dense_6/BiasAdd
functional_7/dense_6/ReluRelu%functional_7/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_7/dense_6/ReluÍ
*functional_7/dense_7/MatMul/ReadVariableOpReadVariableOp3functional_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*functional_7/dense_7/MatMul/ReadVariableOpÓ
functional_7/dense_7/MatMulMatMul'functional_7/dense_6/Relu:activations:02functional_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_7/dense_7/MatMulË
+functional_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4functional_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_7/dense_7/BiasAdd/ReadVariableOpÕ
functional_7/dense_7/BiasAddBiasAdd%functional_7/dense_7/MatMul:product:03functional_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_7/dense_7/BiasAddy
IdentityIdentity%functional_7/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT:::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_4


,__inference_conv2d_10_layer_call_fn_48119918

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_481194692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_48119518

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü
ý
/__inference_functional_7_layer_call_fn_48119721
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_7_layer_call_and_return_conditional_losses_481196982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_4
Ô(
°
J__inference_functional_7_layer_call_and_return_conditional_losses_48119828

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp¿
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/Relu³
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_10/Conv2D/ReadVariableOp×
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
2
conv2d_10/Conv2Dª
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
conv2d_10/Relu³
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_11/Conv2D/ReadVariableOpØ
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_11/Conv2Dª
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_3/Const
flatten_3/ReshapeReshapeconv2d_11/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_3/Reshape§
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
dense_6/MatMul/ReadVariableOp 
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¥
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¢
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Relu¦
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAddl
IdentityIdentitydense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT:::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
	
¯
G__inference_conv2d_11_layer_call_and_return_conditional_losses_48119929

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ		@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
Ô(
°
J__inference_functional_7_layer_call_and_return_conditional_losses_48119788

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp¿
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/Relu³
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_10/Conv2D/ReadVariableOp×
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
2
conv2d_10/Conv2Dª
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@2
conv2d_10/Relu³
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_11/Conv2D/ReadVariableOpØ
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_11/Conv2Dª
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_3/Const
flatten_3/ReshapeReshapeconv2d_11/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_3/Reshape§
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
dense_6/MatMul/ReadVariableOp 
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¥
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¢
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Relu¦
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAddl
IdentityIdentitydense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿTT:::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
input_48
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿTT;
dense_70
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÐÚ
D
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
S_default_save_signature
*T&call_and_return_all_conditional_losses
U__call__"ñ@
_tf_keras_networkÕ@{"class_name": "Functional", "name": "functional_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 84, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_7", 0, 0]]}}, "training_config": {"loss": {"class_name": "Huber", "config": {"reduction": "auto", "name": "huber_loss", "delta": 1.0}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1.0, "learning_rate": 0.00025, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
ù"ö
_tf_keras_input_layerÖ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
ñ	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"Ì
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 84, 4]}}
õ	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 32]}}
ó	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
æ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*\&call_and_return_all_conditional_losses
]__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
õ

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*^&call_and_return_all_conditional_losses
___call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3136}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3136]}}
ó

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*`&call_and_return_all_conditional_losses
a__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
"
	optimizer
f
0
1
2
3
4
5
$6
%7
*8
+9"
trackable_list_wrapper
f
0
1
2
3
4
5
$6
%7
*8
+9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê

0layers
1layer_regularization_losses
		variables
2layer_metrics

trainable_variables
regularization_losses
3non_trainable_variables
4metrics
U__call__
S_default_save_signature
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
,
bserving_default"
signature_map
):' 2conv2d_9/kernel
: 2conv2d_9/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

5layers
6layer_regularization_losses
	variables
7layer_metrics
trainable_variables
regularization_losses
8non_trainable_variables
9metrics
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_10/kernel
:@2conv2d_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

:layers
;layer_regularization_losses
	variables
<layer_metrics
trainable_variables
regularization_losses
=non_trainable_variables
>metrics
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_11/kernel
:@2conv2d_11/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

?layers
@layer_regularization_losses
	variables
Alayer_metrics
trainable_variables
regularization_losses
Bnon_trainable_variables
Cmetrics
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Dlayers
Elayer_regularization_losses
 	variables
Flayer_metrics
!trainable_variables
"regularization_losses
Gnon_trainable_variables
Hmetrics
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
": 
À2dense_6/kernel
:2dense_6/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

Ilayers
Jlayer_regularization_losses
&	variables
Klayer_metrics
'trainable_variables
(regularization_losses
Lnon_trainable_variables
Mmetrics
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_7/kernel
:2dense_7/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

Nlayers
Olayer_regularization_losses
,	variables
Player_metrics
-trainable_variables
.regularization_losses
Qnon_trainable_variables
Rmetrics
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
é2æ
#__inference__wrapped_model_48119427¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_4ÿÿÿÿÿÿÿÿÿTT
ö2ó
J__inference_functional_7_layer_call_and_return_conditional_losses_48119610
J__inference_functional_7_layer_call_and_return_conditional_losses_48119828
J__inference_functional_7_layer_call_and_return_conditional_losses_48119788
J__inference_functional_7_layer_call_and_return_conditional_losses_48119580À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_functional_7_layer_call_fn_48119878
/__inference_functional_7_layer_call_fn_48119721
/__inference_functional_7_layer_call_fn_48119853
/__inference_functional_7_layer_call_fn_48119666À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_conv2d_9_layer_call_and_return_conditional_losses_48119889¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_9_layer_call_fn_48119898¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv2d_10_layer_call_and_return_conditional_losses_48119909¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_conv2d_10_layer_call_fn_48119918¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv2d_11_layer_call_and_return_conditional_losses_48119929¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_conv2d_11_layer_call_fn_48119938¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_flatten_3_layer_call_and_return_conditional_losses_48119944¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_flatten_3_layer_call_fn_48119949¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_6_layer_call_and_return_conditional_losses_48119960¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_6_layer_call_fn_48119969¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_7_layer_call_and_return_conditional_losses_48119979¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_7_layer_call_fn_48119988¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5B3
&__inference_signature_wrapper_48119748input_4 
#__inference__wrapped_model_48119427y
$%*+8¢5
.¢+
)&
input_4ÿÿÿÿÿÿÿÿÿTT
ª "1ª.
,
dense_7!
dense_7ÿÿÿÿÿÿÿÿÿ·
G__inference_conv2d_10_layer_call_and_return_conditional_losses_48119909l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ		@
 
,__inference_conv2d_10_layer_call_fn_48119918_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ		@·
G__inference_conv2d_11_layer_call_and_return_conditional_losses_48119929l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_11_layer_call_fn_48119938_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª " ÿÿÿÿÿÿÿÿÿ@¶
F__inference_conv2d_9_layer_call_and_return_conditional_losses_48119889l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿTT
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
+__inference_conv2d_9_layer_call_fn_48119898_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿTT
ª " ÿÿÿÿÿÿÿÿÿ §
E__inference_dense_6_layer_call_and_return_conditional_losses_48119960^$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_6_layer_call_fn_48119969Q$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_7_layer_call_and_return_conditional_losses_48119979]*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_7_layer_call_fn_48119988P*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
G__inference_flatten_3_layer_call_and_return_conditional_losses_48119944a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 
,__inference_flatten_3_layer_call_fn_48119949T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀÃ
J__inference_functional_7_layer_call_and_return_conditional_losses_48119580u
$%*+@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿTT
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
J__inference_functional_7_layer_call_and_return_conditional_losses_48119610u
$%*+@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿTT
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_functional_7_layer_call_and_return_conditional_losses_48119788t
$%*+?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿTT
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_functional_7_layer_call_and_return_conditional_losses_48119828t
$%*+?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿTT
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_functional_7_layer_call_fn_48119666h
$%*+@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿTT
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_functional_7_layer_call_fn_48119721h
$%*+@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿTT
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_functional_7_layer_call_fn_48119853g
$%*+?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿTT
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_functional_7_layer_call_fn_48119878g
$%*+?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿTT
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¯
&__inference_signature_wrapper_48119748
$%*+C¢@
¢ 
9ª6
4
input_4)&
input_4ÿÿÿÿÿÿÿÿÿTT"1ª.
,
dense_7!
dense_7ÿÿÿÿÿÿÿÿÿ