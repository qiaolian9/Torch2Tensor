import tvm
from tvm import meta_schedule as ms
from tvm import IRModule
from loguru import logger
import os
import shutil

__all__ = ['mlc_tuner']

class mlc_tuner:
    def __init__(self, target='llvm -num-cores 36', work_dir="./tune_tmp", 
            task_name='main', max_trials_global=64, 
            num_trials_per_iter=64, compile_tir_target='llvm'
    ):
        self.target = target
        self.work_dir = work_dir
        self.task_name = task_name
        self.max_trials_global = max_trials_global
        self.num_trials_per_iter = num_trials_per_iter
        self.compile_tir_target = compile_tir_target
        self.pre_trained = False
        logger.info("target: %s; compile_tir_target: %s" % (self.target, self.compile_tir_target))
    
    def mlc_tune_tir(self, Model: IRModule, op_list=None):
        if os.path.exists(self.work_dir):
            if not self.pre_trained:
                shutil.rmtree(self.work_dir)
                os.makedirs(self.work_dir)
        else:
            os.makedirs(self.work_dir)
        if op_list is None:
            raise Warning('please indicate the ops which will be tuned!')
        
        if 'main' in op_list:
            op_list.remove('main')
        logger.info(op_list)
        for i, op_name in enumerate(op_list):
            logger.info("read to tune no.%d op: %s" % (i, op_name))
            mod_ = IRModule.from_expr(Model[op_name].with_attr('global_symbol', 'main'))
            new_func = self.mlc_tune_op(mod_, op_name)
            gv = Model.get_global_var(op_name)
            Model.update_func(gv, new_func)

        return Model

    def mlc_tune_op(self, mod_, op_name):
        logger.info("op: "+ op_name)
        op_work_dir = self.work_dir + '/op_%s' % (op_name)
        try:
            tuned_record = ms.tune_tir(
                mod_, target=self.target,
                work_dir = op_work_dir,
                task_name = self.task_name,
                max_trials_global = self.max_trials_global,
                num_trials_per_iter = self.num_trials_per_iter,
            )
            tuned_sch = ms.tir_integration.compile_tir(tuned_record, mod_, target=self.compile_tir_target)
            new_func = tuned_sch.mod['main'].with_attr('global_symbol', op_name)
            return new_func
        except:
            raise Warning("op %s is not in Model, please check func::print_op(self)" % op_name)
    
    def mlc_load(self, record_path):
        pass


