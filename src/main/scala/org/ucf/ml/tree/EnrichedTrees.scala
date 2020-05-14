package org.ucf.ml
package tree

import scala.collection.JavaConversions._
import com.github.javaparser.ast.body._
import com.github.javaparser.ast._
import org.ucf.ml.Context


trait EnrichedTrees extends _Statement{

  implicit class genCompilationUnit(node:CompilationUnit) {
    def genCode(ctx:Context):Unit = {
      // 1. package declaration
      val package_decl = node.getPackageDeclaration
      if (package_decl.isPresent) package_decl.get().genCode(ctx)

      // 2. Import Statements
      node.getImports.toList.foreach(impl => {
        impl.genCode(ctx)
        ctx.appendNewLine()
      })

      // 3. A list of defined types, such as Class, Interface, Enum, Annotation ...
      node.getTypes.toList.foreach(typeDecl => typeDecl.genCode(ctx))

    }
  }

  implicit class genPackageDeclaration(node: PackageDeclaration) {
    def genCode(ctx:Context):Unit = {
      ctx.append("package")
      node.getName.genCode(ctx)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  /**
   *  import java.io.*;
   *  import static java.io.file.out;
   * @param node
   */
  implicit class genImportDeclaration(node:ImportDeclaration) {
    def genCode(ctx:Context):Unit = {
      ctx.append("import")

      if (node.isStatic) ctx.append("static")

      node.getName.genCode(ctx)

      if (node.isAsterisk) {
        ctx.append(".")
        ctx.append("*")
      }
      ctx.append(";")
    }
  }

  implicit class genTypeDeclaration(node:TypeDeclaration[_]) {
    def genCode(ctx:Context):Unit = {
      node match {
        /**
         *  TypeDeclaration
         *     -- EnumDeclaration
         *     -- AnnotationDeclaration
         *     -- ClassOrInterfaceDeclaration
         */
        case n: EnumDeclaration => {n.genCode(ctx)}
        case n: AnnotationDeclaration => {n.genCode(ctx)}
        case n: ClassOrInterfaceDeclaration => {n.genCode(ctx)}
      }
      
    }
  }

  implicit class genEnumDeclaration(node:EnumDeclaration) {
    def genCode(ctx:Context):Unit = {
      val modifier = node.getModifiers.toList
      modifier.foreach(_.genCode(ctx))

      val name = node.getName
      name.genCode(ctx)
      ctx.append("{")
      val entries = node.getEntries.toList
      entries.foreach(entry => {
        entry.genCode(ctx)
        if (entry != entries.last) ctx.append(",")
      })
      ctx.append("}")
      ctx.appendNewLine()
    }
    implicit class genEnumConstantDeclaration(node:EnumConstantDeclaration) {
      def genCode(ctx:Context):Unit = {
        val name = node.getName
        name.genCode(ctx)
        //TODO  the body is not complete at all
        ctx.append(node.toString)
      }
    }
  }

  implicit class genAnnotationDeclaration(node:AnnotationDeclaration) {
    def genCode(ctx:Context):Unit = {
      //TODO, No implementation about annotation
      ctx.append(node.toString)
    }
  }

  implicit class genClassOrInterfaceDeclaration(node:ClassOrInterfaceDeclaration) {
    def genCode(ctx:Context):Unit = {
      // 1. Class Modifiers, such as public/private
      val modifiers = node.getModifiers.toList
      modifiers.foreach(modifier => modifier.genCode(ctx))

      if (node.isInterface) ctx.append("interface") else ctx.append("class")

      // 2. Interface/Class Name
      ctx.append(ctx.type_maps.getNewContent(node.getNameAsString))

      // 3. type parameters public interface Predicate<T> {}
      val tps = node.getTypeParameters.toList
      tps.foreach(_.genCode(ctx))

      ctx.append("{")
      ctx.appendNewLine()

      // 3. Class Members: Filed and method, constructor
      val members = node.getMembers.toList
      members.foreach(bodyDecl => bodyDecl.genCode(ctx))

      ctx.append("}")
      ctx.appendNewLine()
      
    }
  }

  implicit class genBodyDeclaration(node:BodyDeclaration[_]){
    def genCode(ctx:Context):Unit = {
      //TODO, only implment [[CallableDeclaration]] to handle with Method Declare
      node match {
        case n: InitializerDeclaration => {n.genCode(ctx)}
        case n: FieldDeclaration => {n.genCode(ctx)}
        case n: TypeDeclaration[_] => n.genCode(ctx)
        case n: EnumConstantDeclaration => {n.genCode(ctx)}
        case n: AnnotationMemberDeclaration => {n.genCode(ctx)}
        case n: CallableDeclaration[_] => {n.genCode(ctx)}
      }
    }
  }

  implicit class genInitializerDeclaration(node:InitializerDeclaration) {
    def genCode(ctx:Context):Unit = {
      //TODO, no implmentation for learning bugs
    }
  }

  implicit class genFieldDeclaration(node:FieldDeclaration) {
    def genCode(ctx:Context):Unit = {
      //TODO, no implmentation for learning bugs
      node.getModifiers.toList.foreach(_.genCode(ctx))
      val varibles = node.getVariables.toList


      varibles.foreach(ele => {
        if (ele == varibles.head)
            ele.getType.genCode(ctx)

        ele.getName.genCode(ctx)
        if (ele.getInitializer.isPresent){
          ctx.append("=")
          ele.getInitializer.get().genCode(ctx)
        }

        if (ele != node.getVariables.last) ctx.append(",")
      })
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genEnumConstantDeclaration(node:EnumConstantDeclaration) {
    def genCode(ctx:Context):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString)
    }
  }

  implicit class genAnnotationMemberDeclaration(node:AnnotationMemberDeclaration) {
    def genCode(ctx:Context):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString)
    }
  }

  implicit class genCallableDeclaration(node:CallableDeclaration[_]){
    def genCode(ctx:Context):Unit = {
      node match {
        //TODO, no implmentation [[ConstructorDeclaration]] for learning bugs
        case n: ConstructorDeclaration => {n.genCode(ctx)}
        case n: MethodDeclaration => {n.genCode(ctx)}
      }
      
    }
  }

  implicit class genConstructorDeclaration(node:ConstructorDeclaration) {
    def genCode(ctx:Context):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString)
    }
  }

  implicit class genMethodDeclaration(node:MethodDeclaration) {
    def genCode(ctx:Context):Unit = {

      /*modifiers, such as public*/
      val modifiers = node.getModifiers.toList
      modifiers.foreach(modifier => modifier.genCode(ctx))

      /*Method return type, such as void, int, string*/
      node.getType.genCode(ctx)

      /*Method name, such as hello*/
      val value = ctx.method_maps.getNewContent(node.getName.toString())
      ctx.append(value)

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

    }
  }

}
