package org.ucf.ml
package parallel


import java.util.concurrent.Callable

import scala.collection.mutable


class Worker(src_batch:List[String] = null,
             tgt_batch:List[String] = null,
             idioms:mutable.HashSet[String],
             worker_id:Int) extends Callable[String] with utils.Common{

  import java.util.concurrent.Callable

  val javaPaser = new parser.JavaParser
  val ctx = new Context(idioms)

  val batch_size = scala.math.min(src_batch.size, tgt_batch.size)

  def task(buggyPath:String = "data/1/buggy.java",
           fixedPath:String = "src/main/java/org/ucf/ml/JavaApp.java") = {

    logger.debug(f"Process Buggy Source code ${buggyPath}")
    ctx.setCurrentMode(SOURCE)
    val buggy_cu = javaPaser.getComplationUnit(buggyPath, "method")

    if (logger.isDebugEnabled) javaPaser.printAST("./log/buggy.Yaml", buggy_cu)

    javaPaser.addPositionWithGenCode(ctx, buggy_cu)


    logger.debug(f"Process Fixed Source code ${fixedPath}\n")
    ctx.setCurrentMode(TARGET)
    val fixed_cu = javaPaser.getComplationUnit(fixedPath, "method")

    if (logger.isDebugEnabled) javaPaser.printAST("./log/fixed.Yaml", fixed_cu)

    javaPaser.addPositionWithGenCode(ctx, fixed_cu)



    /*Dumpy buggy and fixed abstract code to a specify file*/

    /*Clear the context and */
    ctx.clear
  }

  override def call(): String = {
    for (idx <- 0 until batch_size) {
      task(src_batch(idx), tgt_batch(idx))
    }
    logger.debug(ctx.get_buggy_abstract)
    logger.debug("********************************************************")
    logger.debug(ctx.get_fixed_abstract)
    EmptyString
  }

}
