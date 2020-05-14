package org.ucf.ml

/**
  * @author 
  */
object ScalaApp {
  def main(args: Array[String]): Unit = {
    /**
     * https://javaparser.org/inspecting-an-ast/
     */
    val ctx = new Context
    val java_parser = new parser.JavaParser(ctx)

    println("******************Buggy Gen Code **************************")
    ctx.setCurrentTarget("buggy")
    val cu_buggy = java_parser.getComplationUnit("data/1/buggy.java", "method")
    java_parser.printAST("./log/buggy.Yaml", cu_buggy)
    java_parser.addPositionWithGenCode(ctx, cu_buggy)
    println(ctx.get_buggy_abstract)


    println("******************Fixed Gen Code **************************")
    ctx.setCurrentTarget("fixed")
    //    val cu_fixed = getComplationUnit("data/1/fixed.java", "method")
    val cu_fixed = java_parser.getComplationUnit("src/main/java/org/ucf/ml/JavaApp.java", "class")
    java_parser.printAST("./log/fixed.Yaml", cu_fixed)
    java_parser.addPositionWithGenCode(ctx, cu_fixed)
    println(ctx.get_fixed_abstract)

    ctx.dumpy_mapping("data/1/mapping.txt")
  }
}