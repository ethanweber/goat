import React, {Component} from 'react';
import './App.css';
import * as THREE from 'three';
import {TrackballControls} from 'three/examples/jsm/controls/TrackballControls';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import {GUI} from 'three/examples/jsm/libs/dat.gui.module.js';
import {split_path} from "./utils";
import {SceneNode} from "./SceneNode";
import {ExtensibleObjectLoader} from "./ExtensibleObjectLoader";

var msgpack = require("msgpack-lite");

export class Viewer extends Component {
    constructor(props) {
        super(props);
        this.state = {
            scene: null,
            controls: null,
            renderer: null,
            camera: null,
            gui: null,
            scene_tree: null,
            needs_render: true
        }
        this.animate = this.animate.bind(this);
        this.set_object = this.set_object.bind(this);
        this.handle_command = this.handle_command.bind(this);
    }

    set_dirty() {
        // TODO(ethan): use this!
        this.state.needs_render = true;
    }

    animate() {
        requestAnimationFrame(this.animate);
        if (this.state.needs_render) {
            this.state.controls.update();
            this.state.renderer.render(this.state.scene, this.state.camera);
            // TODO(ethan): implement this, so rendering doesn't occur all the time
            // this.state.needs_render = false;
        }
    }

    set_object(path, object) {
        this.state.scene_tree.find(path.concat(["<object>"])).set_object(object);
    }

    // set_object_from_json(path, object_json) {
    //     let loader = new THREE.ObjectLoader();
    //     console.log(object_json);
    //     loader.parse(object_json, (obj) => {
    //         console.log(obj);
    //         this.set_object(path, obj);
    //     });
    // }

    set_object_from_json(path, object_json) {
        let loader = new ExtensibleObjectLoader();
        loader.onTextureLoad = () => {
            this.set_dirty();
        }
        loader.parse(object_json, (obj) => {
            if (obj.geometry !== undefined && obj.geometry.type == "BufferGeometry") {
                if ((obj.geometry.attributes.normal === undefined) || obj.geometry.attributes.normal.count === 0) {
                    obj.geometry.computeVertexNormals();
                }
            } else if (obj.type.includes("Camera")) {
                this.set_camera(obj);
                this.set_3d_pane_size();
            }
            obj.castShadow = true;
            obj.receiveShadow = true;
            this.set_object(path, obj);
            this.set_dirty();
        });
    }

    set_transform(path, matrix) {
        this.state.scene_tree.find(path).set_transform(matrix);
    }

    delete_path(path) {
        if (path.length == 0) {
            console.error("Deleting the entire scene is not implemented.")
        } else {
            this.state.scene_tree.delete(path);
        }
    }

    set_property(path, property, value) {
        this.state.scene_tree.find(path).set_property(property, value);
        // TODO(ethan): handle this issue
        if (path[0] === "Background") {
            // The background is not an Object3d, so needs a little help.
            this.state.scene_tree.find(path).on_update();
        }
    }

    handle_command(cmd) {
        console.log("handle_command");
        console.log(cmd);

        // convert binary serialization format back to JSON
        cmd = msgpack.decode(new Uint8Array(cmd));
        console.log(cmd);

        // TODO(ethan): ignore these or remove status. maybe incorporate into a clean view
        if (cmd.type === "status") {
            return;
        }

        if (cmd.type === "set_object") {
            let path = split_path(cmd.path);
            this.set_object_from_json(path, cmd.object);
        } else if (cmd.type === "set_transform") {
            let path = split_path(cmd.path);
            this.set_transform(path, cmd.matrix);
        } else if (cmd.type === "delete") {
            let path = split_path(cmd.path);
            this.delete_path(path);
        } else if (cmd.type === "set_property") {
            let path = split_path(cmd.path);
            this.set_property(path, cmd.property, cmd.value);
        } else if (cmd.type === "set_animation") {
            // TODO(ethan): implement animations
            console.error("Animations not implemented yet.")
        }
        this.set_dirty();
    }

    save_scene() {
    }

    load_scene() {
    }

    save_image() {
    }

    componentDidMount() {
        // Scene
        this.state.scene = new THREE.Scene();
        this.state.scene.background = new THREE.Color(0xFFFFFF);

        // GUI
        this.state.gui = new GUI();
        let scene_folder = this.state.gui.addFolder("Scene");
        scene_folder.open();
        this.state.scene_tree = new SceneNode(this.state.scene, scene_folder, () => this.set_dirty());
        let save_folder = this.state.gui.addFolder("Save / Load / Capture");
        save_folder.add(this, 'save_scene');
        save_folder.add(this, 'load_scene');
        save_folder.add(this, 'save_image');
        this.state.gui.open();

        // Renderer
        this.state.renderer = new THREE.WebGLRenderer({antialias: true});
        this.state.renderer.setPixelRatio(window.devicePixelRatio);
        this.state.renderer.setSize(window.innerWidth, window.innerHeight);
        this.mount.appendChild(this.state.renderer.domElement);

        // Camera settings
        // https://stackoverflow.com/questions/46182845/field-of-view-aspect-ratio-view-matrix-from-projection-matrix-hmd-ost-calib/46195462
        this.state.camera = new THREE.PerspectiveCamera(119.99058885730445, 0.9966887563898866, 0.01, 1000);
        // Controls
        this.state.controls = new TrackballControls(this.state.camera, this.state.renderer.domElement);
        this.state.controls.rotateSpeed = 2.0;
        this.state.controls.zoomSpeed = 0.3;
        this.state.controls.panSpeed = 0.2;
        this.state.controls.staticMoving = true;
        this.state.controls.target.set(0, 0, 0);
        this.state.controls.update();
        this.state.camera.position.x = 5.0;
        this.state.camera.position.y = 5.0;
        this.state.camera.position.z = 5.0;
        this.state.camera.up.set(0, 0, 1);
        this.set_object(["Main Camera"], this.state.camera);

        // Axes display
        let axes = new THREE.AxesHelper(5);
        this.set_object(["Axes"], axes);

        // Grid
        let grid = new THREE.GridHelper(20, 20);
        grid.rotateX(Math.PI / 2);
        this.set_object(["Grid"], grid);

        // Lights
        let color = 0xFFFFFF;
        let intensity = 1;
        let light = new THREE.AmbientLight(color, intensity);
        this.set_object(["Light"], light);
        // this.state.scene.add(light);
        //
        // GUI
        // this.state.gui = new GUI();
        // gui.addColor(new ColorGUIHelper(light, 'color'), 'value').name('color');
        // gui.add(light, 'intensity', 0, 2, 0.01);

        // Stats
        var stats = new Stats();
        var container = document.createElement('div');
        this.mount.appendChild(container);
        container.appendChild(this.state.renderer.domElement);
        container.appendChild(stats.dom);


        // Cube
        // var geometry = new THREE.BoxGeometry(1, 1, 1);
        // var material = new THREE.MeshBasicMaterial({color: 0x00ff00});
        // var cube = new THREE.Mesh(geometry, material);
        // this.state.scene.add(cube);


        var geometry = new THREE.PlaneGeometry(10, 10);
        console.log(geometry);

        this.animate();
    }

    render() {
        return (
            <div ref={ref => (this.mount = ref)}/>
        )
    }
}