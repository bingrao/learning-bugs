package org.ucf.ml
package tree

import scala.collection.JavaConversions._
import com.github.javaparser.ast.body._
import com.github.javaparser.ast._
import utils.{Common, Context}


trait EnrichedTrees extends Common with _Statement{


  implicit class addPosition(node:Node) {
    def getPosition(ctx: utils.Context) = ctx.getNewPosition
  }

  implicit class genCompilationUnit(node:CompilationUnit) {
    def genCode(ctx:Context):String = {
      val package_decl = node.getPackageDeclaration
      if (package_decl.isPresent) package_decl.get().genCode(ctx)
      node.getImports.toList.foreach(impl => impl.genCode(ctx))
      node.getTypes.toList.foreach(tp => tp match {
        case n:ClassOrInterfaceDeclaration => n.genCode(ctx)
        case n:EnumDeclaration => {}
        case n:AnnotationDeclaration => {}
      })
      EMPTY_STRING
    }
  }

  implicit class genPackageDeclaration(node: PackageDeclaration) {
    def genCode(ctx:Context):String = {
      ctx.append(content = node.getName.toString)
      EMPTY_STRING
    }
  }

  implicit class genImportDeclaration(node:ImportDeclaration) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getName.toString)
      EMPTY_STRING
    }
  }

  implicit class genClassOrInterfaceDeclaration(node:ClassOrInterfaceDeclaration) {
    def genCode(ctx:Context):String = {
      val modifiers = node.getModifiers.toList
      modifiers.foreach(modifier => modifier.genCode(ctx))

      val name = node.getName
      ctx.append(name.toString)

      ctx.append("{")
      ctx.append("\n")

      val members = node.getMembers.toList
      members.foreach(member => member match {
        case n:InitializerDeclaration => {}
        case n:FieldDeclaration => {}

        /**
         *  TypeDeclaration
         *     -- EnumDeclaration
         *     -- AnnotationDeclaration
         *     -- ClassOrInterfaceDeclaration
         */
        case n:EnumDeclaration => {}
        case n:AnnotationDeclaration => {}
        case n:ClassOrInterfaceDeclaration => {n.genCode(ctx)}

        case n:EnumConstantDeclaration => {}

        /**
         * CallableDeclaration
         *    -- ConstructorDeclaration
         *    -- MethodDeclaration
         */
        case n:ConstructorDeclaration => {}
        case n:MethodDeclaration => {n.genCode(ctx)}
        case n:AnnotationMemberDeclaration => {}
      })
      ctx.append("}")
      ctx.append("\n")
      EMPTY_STRING
    }
  }

  implicit class genMethodDeclaration(node:MethodDeclaration) {
    def genCode(ctx:Context):String = {
      /*modifiers, such as public*/
      val modifiers = node.getModifiers.toList
      modifiers.foreach(modifier => modifier.genCode(ctx))

      /*Method return type, such as void, int, string*/
      node.getType.genCode(ctx)

      /*Method name, such as hello*/
      val name = node.getName
      ctx.append(name.toString)

      /*formal paramters*/
      ctx.append("(")
      val parameters = node.getParameters.toList
      parameters.foreach(p => {
        p.genCode(ctx)
        if (p != parameters.last) ctx.append(",")
      })
      ctx.append(")")

      /*Method Body*/
      val body = node.getBody
      if (body.isPresent) body.get().genCode(ctx)

      EMPTY_STRING
    }
  }

}
