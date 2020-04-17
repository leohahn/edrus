use gluon::{
    vm::api::{Hole, OpaqueValue},
    ThreadExt,
};
use gluon_codegen::{Getable, VmType};
use serde_derive::{Deserialize, Serialize};

#[derive(Getable, VmType, Debug, Serialize, Deserialize)]
#[gluon(vm_type = "edrus.types.EditorConfig")]
pub struct EditorConfig {
    pub font_scale: f32,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self { font_scale: 24.0 }
    }
}

pub fn get_editor_config(vm: &gluon::RootedThread) -> EditorConfig {
    use gluon::vm::api::FunctionRef;

    let create_editor_config_result = vm.get_global("init.create_editor_config");
    if create_editor_config_result.is_err() {
        EditorConfig::default()
    } else {
        let mut create_editor_config: FunctionRef<fn(()) -> EditorConfig> =
            create_editor_config_result.unwrap();
        create_editor_config.call(()).unwrap()
    }
}

pub fn startup_engine() -> std::io::Result<gluon::RootedThread> {
    // Initialize gluon vm
    let gluon_vm = gluon::new_vm();
    gluon_vm
        .run_expr::<OpaqueValue<&gluon::vm::thread::Thread, Hole>>("std.prelude", r#" import! std.prelude "#)
        .unwrap();

    register_edrus_types(&gluon_vm);

    let config_dir = dirs::config_dir().expect("should find a config dir");
    let edrus_dir = config_dir.join("edrus");

    if std::fs::metadata(&edrus_dir).is_ok() {
        let init_script = edrus_dir.join("init.glu");
        let _ = std::fs::read_to_string(init_script)
            .map(|script_str| {
                println!("loading init.glu");
                gluon_vm
                    .load_script("init.glu", &script_str)
                    .expect("should not fail");
            })
            .map_err(|_| println!("not loading init.glu, since it does not exist"));
    }

    Ok(gluon_vm)
}

fn register_edrus_types(vm: &gluon::RootedThread) {
    use gluon::vm::api::typ::make_source;

    let editor_config_source =
        make_source::<EditorConfig>(&vm).expect("should not fail to create type source");
    vm.load_script("edrus.types", &editor_config_source)
        .expect("failed to load script");
}
