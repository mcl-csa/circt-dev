//===- ESIServices.td - ESI services ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/RegionKindInterface.td"

def ServiceDeclOp : ESI_Op<"service.decl",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">, Symbol]> {
  let summary = "An ESI service interface declaration";
  let description = [{
    A declaration of an ESI service interface. Defines a contract between a
    service provider and its clients.

    Example:

    ```mlir
    esi.service.decl @HostComms {
      esi.service.to_server send : !esi.channel<!esi.any>
      esi.service.to_client recieve : !esi.channel<i8>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$ports);

  let assemblyFormat = [{
    $sym_name $ports attr-dict
  }];
}

def ToServerOp : ESI_Op<"service.to_server",
                        [HasParent<"::circt::esi::ServiceDeclOp">]> {
  let summary = "An ESI service port headed to the service";

  let arguments = (ins SymbolNameAttr:$inner_sym, TypeAttr:$type);
  let assemblyFormat = [{
    $inner_sym attr-dict `:` $type
  }];
}

def ToClientOp : ESI_Op<"service.to_client",
                        [HasParent<"::circt::esi::ServiceDeclOp">]> {
  let summary = "An ESI service port headed to a particular client";

  let arguments = (ins SymbolNameAttr:$inner_sym, TypeAttr:$type);
  let assemblyFormat = [{
    $inner_sym  attr-dict `:` $type
  }];
}

def ServiceInstanceOp : ESI_Op<"service.instance"> {
  let summary = "Instantiate a server module";
  let description = [{
    Instantiate a service adhering to a service declaration interface.

    A pass collects all of the connection requests to the service this op
    implements from the containing modules' descendants (in the instance
    hierarchy). It bubbles them all up to the module containing this op,
    creating the necessary ESI channel ports, groups them appropriately, then
    replaces this op with a `service.impl_req`.

    Since implementing the server will usually need "normal" I/O, `inputs` and
    `results` act like normal `hw.instance` ports.

    $identifier is used by frontends to specify or remember the type of
    implementation to use for this service.
  }];

  let arguments = (ins FlatSymbolRefAttr:$service_symbol,
                       AnyAttr:$impl_type,
                       OptionalAttr<DictionaryAttr>:$impl_opts,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $service_symbol `impl` `as` $impl_type (`opts` $impl_opts^)?
      `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def ServiceImplementReqOp : ESI_Op<"service.impl_req", [NoTerminator]> {
  let summary = "Request for a service to be implemented";
  let description = [{
    The connect services pass replaces `service.instance`s with this op. The
    `portReqs` region is the set of connection requests which need to be
    implemented for this service instance. Channels to/from the requests have
    been added to the operands/results of this op and consumers/producers have
    been redirected.

    Some other pass or frontend is expected to replace this op with an actual
    implementation.
  }];

  let arguments = (ins FlatSymbolRefAttr:$service_symbol,
                       AnyAttr:$impl_type,
                       OptionalAttr<DictionaryAttr>:$impl_opts,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$portReqs);

  let assemblyFormat = [{
    $service_symbol `impl` `as` $impl_type (`opts` $impl_opts^)?
     `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      $portReqs
  }];
}

def RequestToServerConnectionOp : ESI_Op<"service.req.to_server", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Request a connection to send data";

  let arguments = (ins HWInnerRefAttr:$servicePort,
                       ChannelType:$sending, StrArrayAttr:$clientNamePath);
  let assemblyFormat = [{
    $sending `->` $servicePort `(` $clientNamePath `)`
      attr-dict `:` qualified(type($sending))
  }];
}

def RequestToClientConnectionOp : ESI_Op<"service.req.to_client", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Request a connection to receive data";

  let arguments = (ins HWInnerRefAttr:$servicePort,
                       StrArrayAttr:$clientNamePath);
  let results = (outs ChannelType:$receiving);
  let assemblyFormat = [{
    $servicePort `(` $clientNamePath `)`
      attr-dict `:` qualified(type($receiving))
  }];
}
