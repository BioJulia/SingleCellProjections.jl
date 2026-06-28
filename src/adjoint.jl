# NB: We call it transpose even though we use adjoint internally.
#     Because a user is more likely to use data' than transpose(data) even when they mean transposing.
"""
    SCP.transpose(data) -> Job

Transpose a `DataMatrix`, swapping variables and observations.
"""
function transpose(data)
	create_job(DataMatrixFunction(Impl.adjoint), data)
end
