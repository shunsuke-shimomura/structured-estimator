use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(EstimationState)]
pub fn derive_estimation_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let vis = &input.vis;

    // ---------- フィールド列（named struct 限定） ----------
    let fields_named = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    data.struct_token,
                    "EstimationState は named struct にだけ使えます",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input, "EstimationState は struct にだけ使えます")
                .to_compile_error()
                .into();
        }
    };

    // ---------- 各フィールドから sigma 用情報を集める ----------

    // シグマ点 struct 名
    let sigma_name = syn::Ident::new(&format!("{}SigmaPoint", name), name.span());

    // シグマ点 struct のフィールド定義: #vis field: <T as GaussianValueType>::Sigma
    let mut sigma_fields = Vec::new();

    // ノミナル struct 名
    let nominal_name = syn::Ident::new(&format!("{}Nominal", name), name.span());

    // ノミナル struct のフィールド定義： #vis field: <T as GaussianValueType>::Nominal
    let mut nominal_fields = Vec::new();

    // シグマ点とノミナル値からの変換用
    let mut merge_sigma_calls = Vec::new();

    // SVector flatten/unflatten 用
    let mut dim_expr = None::<proc_macro2::TokenStream>;
    let mut write_blocks = Vec::new();
    let mut read_blocks = Vec::new();
    let mut sigma_idents = Vec::new();

    for field in fields_named.iter() {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        // --- シグマ点 struct のフィールド型 ---
        let sigma_ty = quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Sigma };
        sigma_fields.push(quote! {
            #vis #ident: #sigma_ty,
        });

        // --- ノミナル struct のフィールド型 ---
        let nominal_ty =
            quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Nominal };
        nominal_fields.push(quote! {
            #vis #ident: #nominal_ty,
        });

        // --- ノミナル値 + シグマ点 の加算呼び出し ---
        merge_sigma_calls.push(quote! {
            #ident: self.#ident.merge_sigma(&sigma.#ident),
        });

        // --- flatten/unflatten 用 ---
        let sigma_component_dim =
            quote! { <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM };

        dim_expr = Some(match dim_expr {
            None => sigma_component_dim.clone(),
            Some(prev) => quote! { #prev + #sigma_component_dim },
        });

        sigma_idents.push(ident.clone());

        // struct -> SVector
        write_blocks.push(quote! {
            {
                let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::write_to_slice(
                    &value.#ident,
                    &mut data[offset .. offset + dim],
                );
                offset += dim;
            }
        });

        // SVector -> struct
        read_blocks.push(quote! {
            let #ident = {
                let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                let v = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::read_from_slice(
                    &slice[offset .. offset + dim],
                );
                offset += dim;
                v
            };
        });
    }

    let dim_expr =
        dim_expr.expect("EstimationState 対象 struct には 1 つ以上のフィールドが必要です");

    // ---------- 生成コード ----------

    let expanded = quote! {

        // シグマ点構造体
        #[derive(Clone, core::marker::Copy)]
        #vis struct #sigma_name {
            #(#sigma_fields)*
        }

        impl Default for #sigma_name {
            fn default() -> Self {
                Self {
                    #(#sigma_idents: Default::default()),*
                }
            }
        }

        // ノミナル構造体
        #[derive(Clone)]
        #vis struct #nominal_name {
            #(#nominal_fields)*
        }

        impl ::structured_estimator::value_structs::NominalStructTrait for #nominal_name {
            type SigmaStruct = #sigma_name;
            type ValueStruct = #name;
            fn merge_sigma(&self, sigma: &Self::SigmaStruct) -> Self::ValueStruct
            where
                Self: Sized,
            {
                use ::structured_estimator::components::GaussianValueType;
                use ::structured_estimator::components::GaussianNominalType;
                Self::ValueStruct {
                    #(#merge_sigma_calls)*
                }
            }
        }

        impl ::structured_estimator::value_structs::ValueStructTrait for #name {
            type SigmaStruct = #sigma_name;
            type NominalStruct = #nominal_name;

            fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct)
            where
                Self: Sized,
            {
                use ::structured_estimator::components::GaussianValueType;
                let mut nominal = Self::NominalStruct {
                    #(#sigma_idents: Default::default()),*
                };
                let mut sigma = Self::SigmaStruct {
                    #(#sigma_idents: Default::default()),*
                };
                #(
                    (nominal.#sigma_idents, sigma.#sigma_idents) = self.#sigma_idents.algebraize();
                )*;
                (nominal, sigma)
            }
        }

        impl ::structured_estimator::value_structs::OutputStructTrait<{ #dim_expr }> for #name {
            fn error_from(
                &self,
                measured: &Self,
            ) -> Self::SigmaStruct {
                use ::structured_estimator::components::GaussianValueType;
                let mut error = Self::SigmaStruct {
                    #(#sigma_idents: Default::default()),*
                };
                #(
                    error.#sigma_idents = self.#sigma_idents.error(&measured.#sigma_idents);
                )*;
                error
            }
        }

        impl ::structured_estimator::value_structs::StateStructTrait<{ #dim_expr }> for #name {

            fn to_sigma(&self, covariance: ::nalgebra::SMatrix::<f64, { #dim_expr }, { #dim_expr }>) -> Result<::structured_estimator::value_structs::SigmaPointSetWithoutWeight<Self, { #dim_expr }>, ::structured_estimator::components::KalmanFilterError>
            where
                Self: Sized,
            {
                ::structured_estimator::ukf::generate_sigma_points(self, covariance)
            }
        }

        impl #sigma_name {
            #vis const DIM: usize = #dim_expr;
        }

        // struct -> SVector 変換
        impl From<#sigma_name> for ::nalgebra::SVector<f64, { #dim_expr }> {
            fn from(value: #sigma_name) -> Self {
                let mut data = [0.0_f64; #dim_expr];
                let mut offset = 0usize;

                #(#write_blocks)*

                ::nalgebra::SVector::<f64, { #dim_expr }>::from_row_slice(&data)
            }
        }

        // SVector -> struct 変換
        impl From<::nalgebra::SVector<f64, { #dim_expr }>> for #sigma_name {
            fn from(vec: ::nalgebra::SVector<f64, { #dim_expr }>) -> Self {
                let slice = vec.as_slice();
                let mut offset = 0usize;

                #(#read_blocks)*

                Self {
                    #(#sigma_idents),*
                }
            }
        }
    };

    expanded.into()
}

#[proc_macro_derive(EstimationGaussianInput, attributes(group))]
pub fn derive_estimation_gaussian_input(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let vis = &input.vis;

    // ---------- フィールド列（named struct 限定） ----------
    let fields_named = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    data.struct_token,
                    "EstimationInput は named struct にだけ使えます",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input, "EstimationInput は struct にだけ使えます")
                .to_compile_error()
                .into();
        }
    };

    // ---------- 各フィールドから sigma/gaussian 用情報を集める ----------

    // シグマ点 struct 名
    let sigma_name = syn::Ident::new(&format!("{}SigmaPoint", name), name.span());

    // ノミナル struct 名
    let nominal_name = syn::Ident::new(&format!("{}Nominal", name), name.span());

    // ガウシアン struct 名
    let gaussian_name = syn::Ident::new(&format!("{}Gaussian", name), name.span());

    // シグマ点 struct のフィールド定義: #vis field: <T as GaussianValueType>::Sigma
    let mut sigma_fields = Vec::new();

    // ノミナル struct のフィールド定義： #vis field: <T as GaussianValueType>::Nominal
    let mut nominal_fields = Vec::new();

    // 代数化用の呼び出し
    let mut algebraize_calls = Vec::new();

    // ガウシアン struct のフィールド定義: #vis field: T, #vis field_covariance: <T as GaussianComponent>::Cov
    let mut gaussian_fields = Vec::new();

    // ガウシアンの結合共分散の配置用コード
    let mut gaussian_cov_write_blocks = Vec::new();

    // SVector flatten/unflatten 用
    let mut dim_expr = None::<proc_macro2::TokenStream>;
    let mut write_blocks = Vec::new();
    let mut read_blocks = Vec::new();
    let mut sigma_idents = Vec::new();

    // シグマ点とノミナル値の変換用
    let mut merge_sigma_calls = Vec::new();

    // グループ情報を保持する構造体
    struct FieldInfo {
        ident: syn::Ident,
        ty: syn::Type,
        group: Option<String>,
    }

    let mut field_infos = Vec::new();

    // フィールド情報とグループ名を収集
    for field in fields_named.iter() {
        let ident = field.ident.as_ref().unwrap().clone();
        let ty = field.ty.clone();

        // #[group("name")] アトリビュートを探す
        let mut group = None;
        for attr in &field.attrs {
            if attr.path().is_ident("group") {
                match attr.parse_args::<syn::LitStr>() {
                    Ok(lit) => {
                        group = Some(lit.value());
                    }
                    Err(e) => return e.to_compile_error().into(),
                }
            }
        }

        field_infos.push(FieldInfo { ident, ty, group });
    }

    // グループごとにフィールドを整理
    use std::collections::HashMap;
    let mut groups: HashMap<Option<String>, Vec<&FieldInfo>> = HashMap::new();
    for field_info in field_infos.iter() {
        groups
            .entry(field_info.group.clone())
            .or_default()
            .push(field_info);
    }

    for field_info in field_infos.iter() {
        let ident = &field_info.ident;
        let ty = &field_info.ty;

        // --- シグマ点 struct のフィールド型 ---
        let sigma_ty = quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Sigma };
        sigma_fields.push(quote! {
            #vis #ident: #sigma_ty,
        });

        // --- flatten/unflatten 用 ---
        let sigma_component_dim =
            quote! { <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM };

        // --- ノミナル struct のフィールド型 ---
        let nominal_ty =
            quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Nominal };
        nominal_fields.push(quote! {
            #vis #ident: #nominal_ty,
        });

        // --- ノミナル値 + シグマ点 の加算呼び出し ---
        merge_sigma_calls.push(quote! {
            #ident: self.#ident.merge_sigma(&sigma.#ident),
        });

        // 代数化呼び出し
        algebraize_calls.push(quote! {
            (nominal.#ident, sigma.#ident) = self.#ident.algebraize();
        });

        dim_expr = Some(match dim_expr {
            None => sigma_component_dim.clone(),
            Some(prev) => quote! { #prev + #sigma_component_dim },
        });

        sigma_idents.push(ident.clone());

        // struct -> SVector
        write_blocks.push(quote! {
            {
                let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::write_to_slice(
                    &value.#ident,
                    &mut data[offset .. offset + dim],
                );
                offset += dim;
            }
        });

        // SVector -> struct
        read_blocks.push(quote! {
            let #ident = {
                let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                let v = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::read_from_slice(
                    &slice[offset .. offset + dim],
                );
                offset += dim;
                v
            };
        });
    }
    let dim_expr =
        dim_expr.expect("EstimationGaussianInput 対象 struct には 1 つ以上のフィールドが必要です");

    // Gaussian構造体の値フィールドを先に追加
    for field_info in field_infos.iter() {
        let ident = &field_info.ident;
        let ty = &field_info.ty;
        gaussian_fields.push(quote! {
            #vis #ident: #ty,
        });
    }

    // グループごとにGaussian構造体の共分散フィールドと書き込みロジックを生成
    for (group_name, fields) in groups.iter() {
        if let Some(group) = group_name {
            // グループがある場合: グループ全体で1つの共分散行列
            let group_ident = format_ident!("{}_covariance", group);

            // グループ内の全フィールドの次元を合計
            let mut group_dim_expr = None::<proc_macro2::TokenStream>;
            for field_info in fields.iter() {
                let ty = &field_info.ty;
                let sigma_ty =
                    quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Sigma };
                let sigma_component_dim =
                    quote! { <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM };
                group_dim_expr = Some(match group_dim_expr {
                    None => sigma_component_dim.clone(),
                    Some(prev) => quote! { #prev + #sigma_component_dim },
                });
            }
            let group_dim = group_dim_expr.unwrap();

            // Gaussian構造体に共分散フィールドを追加
            gaussian_fields.push(quote! {
                #vis #group_ident: ::nalgebra::SMatrix::<f64, { #group_dim }, { #group_dim }>,
            });

            // 共分散書き込みロジック
            gaussian_cov_write_blocks.push(quote! {
                {
                    let dim = #group_dim;
                    covariance
                        .fixed_slice_mut::<{ #group_dim }, { #group_dim }>(offset, offset)
                        .copy_from(&self.#group_ident);
                    offset += dim;
                }
            });
        } else {
            // グループがない場合: 各フィールドごとに共分散行列
            for field_info in fields.iter() {
                let ident = &field_info.ident;
                let ty = &field_info.ty;
                let sigma_ty =
                    quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Sigma };
                let sigma_component_dim =
                    quote! { <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM };

                let cov_ident = format_ident!("{}_covariance", ident);
                gaussian_fields.push(quote! {
                    #vis #cov_ident: ::nalgebra::SMatrix::<f64, { #sigma_component_dim }, { #sigma_component_dim }>,
                });

                gaussian_cov_write_blocks.push(quote! {
                    {
                        let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                        covariance
                            .fixed_slice_mut::<{ #sigma_component_dim }, { #sigma_component_dim }>(offset, offset)
                            .copy_from(&self.#cov_ident);
                        offset += dim;
                    }
                });
            }
        }
    }

    // ---------- 生成コード ----------

    let expanded = quote! {

        // シグマ点構造体
        #[derive(Clone, core::marker::Copy)]
        #vis struct #sigma_name {
            #(#sigma_fields)*
        }

        impl Default for #sigma_name {
            fn default() -> Self {
                Self {
                    #(#sigma_idents: Default::default()),*
                }
            }
        }

        // ノミナル構造体
        #[derive(Clone)]
        #vis struct #nominal_name {
            #(#nominal_fields)*
        }

        impl ::structured_estimator::value_structs::NominalStructTrait for #nominal_name {
            type ValueStruct = #name;
            type SigmaStruct = #sigma_name;
            fn merge_sigma(&self, sigma: &Self::SigmaStruct) -> Self::ValueStruct
            where
                Self: Sized,
            {
                use ::structured_estimator::components::GaussianValueType;
                use ::structured_estimator::components::GaussianNominalType;
                Self::ValueStruct {
                    #(#merge_sigma_calls)*
                }
            }
        }

        // ガウシアン構造体
        #[derive(Clone)]
        #vis struct #gaussian_name {
            #(#gaussian_fields)*
        }

        impl ::structured_estimator::value_structs::ValueStructTrait for #name {
            type SigmaStruct = #sigma_name;
            type NominalStruct = #nominal_name;

            fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct)
            where
                Self: Sized,
            {
                use ::structured_estimator::components::GaussianValueType;
                let mut nominal = Self::NominalStruct {
                    #(#sigma_idents: Default::default()),*
                };
                let mut sigma = Self::SigmaStruct {
                    #(#sigma_idents: Default::default()),*
                };
                #(#algebraize_calls)*;
                (nominal, sigma)
            }
        }

        impl #gaussian_name {
            #vis const DIM: usize = #dim_expr;
        }

        impl From<#gaussian_name> for #name {
            fn from(value: #gaussian_name) -> Self {
                Self {
                    #(#sigma_idents: value.#sigma_idents),*
                }
            }
        }

        impl ::structured_estimator::value_structs::GaussianInputTrait<{ #dim_expr }> for #gaussian_name {
            type SigmaStruct = #sigma_name;
            type MeanStruct = #name;
            fn to_sigma(&self) -> Result<::structured_estimator::value_structs::SigmaPointSetWithoutWeight<Self::MeanStruct, { #dim_expr }>, ::structured_estimator::components::KalmanFilterError>
            where
                Self: Sized
            {
                let mut covariance =
                    ::nalgebra::SMatrix::<f64, { #dim_expr }, { #dim_expr }>::zeros();
                let mut offset = 0usize;
                // ==== covariance の書き込み ====
                {
                    #(#gaussian_cov_write_blocks)*
                }
                let sqrt_cov = covariance
                    .cholesky()
                    .ok_or(::structured_estimator::components::KalmanFilterError::CholeskyDecompositionFailed)?
                    .l();
                let mean = self.mean();
                ::structured_estimator::ukf::generate_sigma_points_from_sqrt_covariance(&mean, sqrt_cov)
            }
            fn mean(&self) -> Self::MeanStruct
            where
                Self: Sized
            {
                Self::MeanStruct {
                    #(#sigma_idents: self.#sigma_idents.clone()),*
                }
            }
        }

        // struct -> SVector 変換
        impl From<#sigma_name> for ::nalgebra::SVector<f64, { #dim_expr }> {
            fn from(value: #sigma_name) -> Self {
                let mut data = [0.0_f64; #dim_expr];
                let mut offset = 0usize;

                #(#write_blocks)*

                ::nalgebra::SVector::<f64, { #dim_expr }>::from_row_slice(&data)
            }
        }

        // SVector -> struct 変換
        impl From<::nalgebra::SVector<f64, { #dim_expr }>> for #sigma_name {
            fn from(vec: ::nalgebra::SVector<f64, { #dim_expr }>) -> Self {
                let slice = vec.as_slice();
                let mut offset = 0usize;

                #(#read_blocks)*

                Self {
                    #(#sigma_idents),*
                }
            }
        }
    };

    expanded.into()
}

#[proc_macro_derive(EstimationOutputStruct)]
pub fn derive_estimation_output_struct(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let vis = &input.vis;

    // ---------- フィールド列（named struct 限定） ----------
    let fields_named = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    data.struct_token,
                    "EstimationState は named struct にだけ使えます",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input, "EstimationState は struct にだけ使えます")
                .to_compile_error()
                .into();
        }
    };

    // ---------- 各フィールドから sigma 用情報を集める ----------

    // シグマ点 struct 名
    let sigma_name = syn::Ident::new(&format!("{}SigmaPoint", name), name.span());

    // シグマ点 struct のフィールド定義: #vis field: <T as GaussianValueType>::Sigma
    let mut sigma_fields = Vec::new();

    // ノミナル struct 名
    let nominal_name = syn::Ident::new(&format!("{}Nominal", name), name.span());

    // ノミナル struct のフィールド定義： #vis field: <T as GaussianValueType>::Nominal
    let mut nominal_fields = Vec::new();

    // シグマ点とノミナル値からの変換用
    let mut merge_sigma_calls = Vec::new();

    // SVector flatten/unflatten 用
    let mut dim_expr = None::<proc_macro2::TokenStream>;
    let mut write_blocks = Vec::new();
    let mut read_blocks = Vec::new();
    let mut sigma_idents = Vec::new();

    for field in fields_named.iter() {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        // --- シグマ点 struct のフィールド型 ---
        let sigma_ty = quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Sigma };
        sigma_fields.push(quote! {
            #vis #ident: #sigma_ty,
        });

        // --- ノミナル struct のフィールド型 ---
        let nominal_ty =
            quote! { <#ty as ::structured_estimator::components::GaussianValueType>::Nominal };
        nominal_fields.push(quote! {
            #vis #ident: #nominal_ty,
        });

        // --- ノミナル値 + シグマ点 の加算呼び出し ---
        merge_sigma_calls.push(quote! {
            #ident: self.#ident.merge_sigma(&sigma.#ident),
        });

        // --- flatten/unflatten 用 ---
        let sigma_component_dim =
            quote! { <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM };

        dim_expr = Some(match dim_expr {
            None => sigma_component_dim.clone(),
            Some(prev) => quote! { #prev + #sigma_component_dim },
        });

        sigma_idents.push(ident.clone());

        // struct -> SVector
        write_blocks.push(quote! {
            {
                let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::write_to_slice(
                    &value.#ident,
                    &mut data[offset .. offset + dim],
                );
                offset += dim;
            }
        });

        // SVector -> struct
        read_blocks.push(quote! {
            let #ident = {
                let dim = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::DIM;
                let v = <#sigma_ty as ::structured_estimator::components::GaussianSigmaType>::read_from_slice(
                    &slice[offset .. offset + dim],
                );
                offset += dim;
                v
            };
        });
    }
    let dim_expr =
        dim_expr.expect("EstimationOutputStruct 対象 struct には 1 つ以上のフィールドが必要です");

    let expanded = quote! {
        // シグマ点構造体
        #[derive(Clone, core::marker::Copy)]
        #vis struct #sigma_name {
            #(#sigma_fields)*
        }

        impl Default for #sigma_name {
            fn default() -> Self {
                Self {
                    #(#sigma_idents: Default::default()),*
                }
            }
        }

        // ノミナル構造体
        #[derive(Clone)]
        #vis struct #nominal_name {
            #(#nominal_fields)*
        }

        impl ::structured_estimator::value_structs::NominalStructTrait for #nominal_name {
            type SigmaStruct = #sigma_name;
            type ValueStruct = #name;
            fn merge_sigma(&self, sigma: &Self::SigmaStruct) -> Self::ValueStruct
            where
                Self: Sized,
            {
                use ::structured_estimator::components::GaussianValueType;
                use ::structured_estimator::components::GaussianNominalType;
                Self::ValueStruct {
                    #(#merge_sigma_calls)*
                }
            }
        }

        impl ::structured_estimator::value_structs::ValueStructTrait for #name {
            type SigmaStruct = #sigma_name;
            type NominalStruct = #nominal_name;

            fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct)
            where
                Self: Sized,
            {
                use ::structured_estimator::components::GaussianValueType;
                let mut nominal = Self::NominalStruct {
                    #(#sigma_idents: Default::default()),*
                };
                let mut sigma = Self::SigmaStruct {
                    #(#sigma_idents: Default::default()),*
                };
                #(
                    (nominal.#sigma_idents, sigma.#sigma_idents) = self.#sigma_idents.algebraize();
                )*;
                (nominal, sigma)
            }
        }

        impl ::structured_estimator::value_structs::OutputStructTrait<{ #dim_expr }> for #name {
            fn error_from(
                &self,
                measured: &Self,
            ) -> Self::SigmaStruct {
                use ::structured_estimator::components::GaussianValueType;
                let mut error = Self::SigmaStruct {
                    #(#sigma_idents: Default::default()),*
                };
                #(
                    error.#sigma_idents = self.#sigma_idents.error(&measured.#sigma_idents);
                )*;
                error
            }
        }

        // struct -> SVector 変換
        impl From<#sigma_name> for ::nalgebra::SVector<f64, { #dim_expr }> {
            fn from(value: #sigma_name) -> Self {
                let mut data = [0.0_f64; #dim_expr];
                let mut offset = 0usize;

                #(#write_blocks)*

                ::nalgebra::SVector::<f64, { #dim_expr }>::from_row_slice(&data)
            }
        }

        // SVector -> struct 変換
        impl From<::nalgebra::SVector<f64, { #dim_expr }>> for #sigma_name {
            fn from(vec: ::nalgebra::SVector<f64, { #dim_expr }>) -> Self {
                let slice = vec.as_slice();
                let mut offset = 0usize;

                #(#read_blocks)*

                Self {
                    #(#sigma_idents),*
                }
            }
        }
    };

    expanded.into()
}
