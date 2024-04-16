// CRRL (Conservative Rational perceptual chunking + Reinforcement Learning)


//TODO:
//	expectedReward can do funny things (currently commented out, but could be bio-realistic)
//		actionUtilities for cobined chunks turns negative even tho each component is positive
//			{"reward":1.37,"action":"left","chunk":"b,d","before":0.016,"after":-0.009}
//			{"reward":1.37,"action":"left","chunk":"c,a","before":0.077,"after":0.052}
//			{"reward":1.161,"action":"left","chunk":"c,d,a,b","before":0.062,"after":0.055}
//	why is reward passed to learn higher than what was perceived?
//	logging seems too verbose w chunk json dumps. maybe just log chunknames?
//	ugm -> <0 >0 should instead be a separate discrete param for reward default null/0
//	need a non-continuous version (no recency, just frequency)
//		perhaps two diff versions made for repeating and non-repeating learning sets,
//			OR just enable a mechanism to repeat a learning set:
//				
//	create separate classes, overriding methods? or keep parameterized?



'use strict';



const MIN_ACTIVATION=.01;
const NORM_SAMPLE=function(){
	const SAMPLE_SIZE=100000;
	function randnorm(){
		var u=0,v=0;
		while(u===0)u=Math.random(); //Converting [0,1) to (0,1)
		while(v===0)v=Math.random();
		return Math.sqrt(-2.0*Math.log(u))*Math.cos(2.0*Math.PI*v);
	}
	var sample=new Float32Array(SAMPLE_SIZE);
	for(var i=0;i<SAMPLE_SIZE;i++)sample[i]=randnorm();
	sample.rand=()=>sample[Math.floor(Math.random()*SAMPLE_SIZE)];
	return sample;
}();


const U=2,WHY=4,RL=8,CL=16;

var agents=[];


Array.prototype.last=function(){return this[this.length-1]};
Number.prototype.pad=function(p,c=0){return String(this).padStart(p,c);};
Number.prototype.to3dec=function(){return Math.round(this*1000)/1000;};
Number.prototype.to2dec=function(){return Math.round(this*100)/100;};
Number.prototype.toJSON=Number.prototype.to3dec;
Date.prototype.stamp=function(){return this.getFullYear()+(this.getMonth()+1).pad(2)+this.getDate().pad(2)+this.getHours().pad(2)+this.getMinutes().pad(2)+this.getSeconds().pad(2);};
JSON.toJSON=(data,spacing)=>JSON.stringify(data,(k,v)=>(v!==undefined&&v.toJSON)?v.toJSON():v,spacing);
function PASS(){}


class Chunk{
	constructor(chunkID){
		if(chunkID.constructor===Array){
			this.parent1=chunkID[0];
			this.parent2=chunkID[1];
			this.parent1.subchunks.push(this);
			this.parent2.subchunks.push(this);
			this.id=this.parent1.id+','+this.parent2.id;
		}else{
			this.id=chunkID;
		}
		this.activationTimes=[];
		this.actionUtilities={};
		this.subchunks=[];
		this.subchunks.toJSON=function(){return this.map(x=>x.id)};
	}
	toJSON(){return {u:this.actionUtilities,s:this.subchunks}}
	toString(){return this.id;}
	allowActivation(currentTic){
		return (this.parent1.activationTimes.length &&
			this.parent2.activationTimes.length &&
			this.parent1.activationTimes.last()==currentTic &&
			this.parent2.activationTimes.last()==currentTic);
	}
}
class CRRL{
	constructor(allowedActions=[],params={},name='a',verbose=0){
		agents.push(this);
		this.VERSION='CRRL1.0';
		this.name=name;
		this.creationDate=new Date();
		var agent=this;
		this.params=new Proxy({
			crt:2,		// Chunk Retrieval Threshold
			eml:1000,	// Episodic Memory Length
			tdc:.5,		// Temporal Decay
			ugm:.7,		// Utility q-learning discount factor, GaMma; if ugm<0, reward default is null (rather than 0)
			ugs:.05,	// Utility Gain noiSe
			ulr:.1,		// Utility Learning Rate
			ucl:1,		// Utility CaLculation (0:sum, 1:avg, 2:max)
		},{
			get:function(params,param){return params[param];},
			set:function(params,param,val){
				if(param in params){
					params[param]=val;
					if(param=='eml' || param=='tdc'){
						agent.setActivationDecay();
					}else if(param=='ucl'){
						agent.updateActionUtility=
							(val==2)?
								agent.actionUtilityMax
								:agent.actionUtilitySum;
					}else if(param=='ulr'){
						agent.updateChunkActionUtility=
							(val>0)?
								agent.updateU1
								:agent.updateU2;
					}
					return true;
				}else{
					throw "Invalid property "+param;
				}
			}
		});
		Object.assign(this.params,params);
		if(!('eml' in params || 'tdc' in params))this.setActivationDecay();
		if(!('ucl' in params))this.params.ucl=this.params.ucl;
		if(!('ulr' in params))this.params.ulr=this.params.ulr;
		this.chunks={};
		this.restrictActions(allowedActions);
		this.actionUtilities={};
		this.ticStateAction=[];
		this.ticStateAction.toJSON=function(){return this.map(tsa=>[tsa[0],tsa[1].map(c=>c.id),tsa[2]]);};
		this.currentTic=-1;
		this.learning=true;
		this.expectedReward=0;
		this.verbose(verbose);
		this.log({name:this.name,framework:this.VERSION,params:this.params,actions:this.allowedActions});
	}
	end(){	//cleanup code
		this.log({tics:this.currentTic,realExecTime:(new Date())-this.creationDate,chunkNum:Object.keys(this.chunks).length});
		try{this.log(process.memoryUsage());}catch(e){}
		this.log({ltm:this.chunks});
		this.log({episodicMemory:this.ticStateAction});
		// if(this.logRL===PASS)this.logCL({activationDecay:this.activationDecay});
		// else this.logRL({activationDecay:this.activationDecay});
		if(this.logfileStream)this.logfileStream.end();
		var myIndex=agents.indexOf(this);
		if(myIndex>-1)agents.splice(myIndex,1);
	}
	verbose(level){
		if(level){
			//initialize logfile path
			if(!this.writeToLog){
				try{
					var fs = require('fs');
					var lognum=0;
					while(true){
						this.logPath=`${this.VERSION}.${this.name}.${this.creationDate.stamp()}.${lognum}.log`;
						try{
							fs.writeFileSync(this.logPath,'',{flag:'wx'});
							break;
						}catch(e){
							if(e.code=='EEXIST')lognum++;
							else throw e;
						}
					}
					this.logfileStream=fs.createWriteStream(this.logPath,{flags:'a'});
					this.writeToLog=function(x){
						this.logfileStream.write(`${(new Date()).getTime()}\t${JSON.toJSON(x)}\n`);
					}
				}catch(e){
					if(e instanceof ReferenceError){
						this.writeToLog=function(x){
							console.log(`${(new Date()).getTime()}\t${JSON.toJSON(x)}`);
						}
					}else throw e;
				}
			}
			this.log=(level&1)?this.writeToLog:PASS;
			this.logU=(level&U)?this.writeToLog:PASS;
			this.logWhy=(level&WHY)?this.writeToLog:PASS;
			this.logRL=(level&RL)?this.writeToLog:PASS;
			this.logCL=(level&CL)?this.writeToLog:PASS;
		}else{
			this.log=this.logU=this.logRL=this.logCL=this.logWhy=PASS;
		}
	}
	restrictActions(actions){this.allowedActions=actions;}
	calcActivationDecay(t){return Math.pow(t+1,-this.params.tdc);}
	setActivationDecay(){
		var activationDecay=[];
		for(var i=0,a=this.calcActivationDecay(0);a>MIN_ACTIVATION && i<this.params.eml;a=this.calcActivationDecay(++i))
			activationDecay.push(a);
		this.activationDecay=new Float32Array(activationDecay);
	}
	actionUtilitySum(au=0,cau=0){return au+cau;}
	actionUtilityMax(au=0,cau=0){return Math.max(au,cau);}
	updateU1(chunk,action,ticsSinceActivation,reward){
		var cau=chunk.actionUtilities[action]||0;
		chunk.actionUtilities[action]=cau+this.activationDecay[ticsSinceActivation]*this.params.ulr*(reward-cau);
		// chunk.actionUtilities[action]=cau+this.activationDecay[ticsSinceActivation]*this.params.ulr*(reward-(this.expectedReward||0));
		this.logRL({reward:reward,action:action,chunk:chunk.id,before:cau.to3dec(),after:chunk.actionUtilities[action]});
	}
	updateU2(chunk,action,ticsSinceActivation,reward){
		var cau=chunk.actionUtilities[action]||0;
		chunk.actionUtilities[action]=cau-this.params.ulr*(this.activationDecay[ticsSinceActivation]*reward-cau);
		this.logRL({reward:reward,action:action,chunk:chunk.id,before:cau.to3dec(),after:chunk.actionUtilities[action]});
	}
	//TODO: consider: wake does not reset chunk.activationTimes (resets EM for rl but not cl)
	//	maybe have wake just set a lastWoken marker, and use that to set removeSize?
	wake(){this.ticStateAction=[];}
	chunkActivate(chunk){
		if(chunk.activationTimes.length==0||chunk.activationTimes.last()!=this.currentTic){ //ensure one activation per tic
			//activate chunk
			this.state.push(chunk);
			chunk._mostSpecific=true;
			chunk.activationTimes.push(this.currentTic);
			//pass activation to subchunks
			var subchunk;
			for(var i=0;i<chunk.subchunks.length;i++){
				subchunk=chunk.subchunks[i];
				if(subchunk.allowActivation(this.currentTic)){
					subchunk.parent1._mostSpecific=false;
					subchunk.parent2._mostSpecific=false;
					this.chunkActivate(subchunk);
				}
			}
			//update action utilities
			var action,cau;
			for(var i=0;i<this.allowedActions.length;i++){
				action=this.allowedActions[i];
				cau=chunk.actionUtilities[action]||0;
				this.actionUtilities[action]=this.updateActionUtility(this.actionUtilities[action],cau);
				if(cau)
					this.logWhy({chunk:chunk.id,action:action,caU:cau});
			}
		}
	}
	step(perceptualCues,reward=0,forceAction,modelTracing){
		//TODO: move the line below down, so that agent gets to learn even when there's no move
		if(forceAction===undefined&&this.allowedActions.length==0)return "0";
		this.currentTic++;
		this.log({tic:this.currentTic});
		//initialize state and action utilities
		this.state=[];
		this.actionUtilities={};
		//spread chunk activation (i.e. perception)
		var chunk,chunkName;
		for(var i=0;i<perceptualCues.length;i++){
			chunkName=perceptualCues[i];
			chunk=this.chunks[chunkName]||(this.chunks[chunkName]=new Chunk(chunkName));
			this.chunkActivate(chunk);
		}
		//action selection
		var myAction;
		if(modelTracing||forceAction===undefined){
			var maxUtility=-Number.MAX_VALUE,
				currentUtility,action;
			for(var i=0;i<this.allowedActions.length;i++){
				action=this.allowedActions[i];
				if(this.params.ucl==1)this.actionUtilities[action]/=this.state.length;
				if(this.params.ugs){
					currentUtility=this.actionUtilities[action]+NORM_SAMPLE.rand()*this.params.ugs;
					this.logU({action:action,U:this.actionUtilities[action],noisyU:currentUtility});
				}else{
					currentUtility=this.actionUtilities[action];
					this.logU({action:action,U:currentUtility});
				}
				if(currentUtility>maxUtility){
					myAction=action;
					maxUtility=currentUtility;
				}
			}
		}else{
			myAction=forceAction;
		}
		if(forceAction===undefined){
			forceAction=myAction;
		}else if(this.allowedActions.indexOf(forceAction)===-1){
			this.allowedActions.push(forceAction);
		}
		//learning
		if(this.learning){
			//learn new chunks
			var i,j,ticsSinceActivation,chunk,
				chunkActivation,maxActivation1=-100,maxActivation2=-100,
				bestChunk1,bestChunk2,removeSize=-1;
			for(j=0;j<this.state.length;j++){
				chunk=this.state[j];
				if(chunk._mostSpecific){
					//calculating activation for chunk (recency * frequency)
					chunkActivation=Math.random()*MIN_ACTIVATION;
					for(i=0;i<chunk.activationTimes.length;i++){
						ticsSinceActivation=this.currentTic-chunk.activationTimes[i];
						if(ticsSinceActivation>this.activationDecay.length){
							removeSize=i;
						}else{
							chunkActivation+=this.activationDecay[ticsSinceActivation];
						}
					}
					this.logCL({chunk:chunk.id,chunkActivation:chunkActivation,activationTimes:chunk.activationTimes});
					//remove outdated activation times
					chunk.activationTimes.splice(0,removeSize+1);
					//pick 2 most active chunks
					if(chunkActivation>this.params.crt){
						if(chunkActivation>maxActivation1){
							bestChunk2=bestChunk1;
							maxActivation2=maxActivation1;
							bestChunk1=chunk;
							maxActivation1=chunkActivation;
						}else if(chunkActivation>maxActivation2){
							bestChunk2=chunk;
							maxActivation2=chunkActivation;
						}
					}
				}
			}
			//create new child from two most active parents
			if(bestChunk2!==undefined){
				var chunk=new Chunk([bestChunk1,bestChunk2]);
				this.chunks[chunk.id]=chunk;
				this.logCL({newChunk:chunk.id});
			}
			//propagate reward
			if(this.params.ugm>=0||reward!=0){
				var tsa,caUpdated=new Set(),chunkAction;
				removeSize=-1;
				if(this.params.ugm>0)reward+=this.params.ugm*(this.actionUtilities[forceAction]||0);
				for(i=0;i<this.ticStateAction.length;i++){
					tsa=this.ticStateAction[i];
					ticsSinceActivation=this.currentTic-tsa[0];
					if(ticsSinceActivation>=this.activationDecay.length){
						removeSize=i;
					}else{
						for(j=0;j<tsa[1].length;j++){
							chunk=tsa[1][j];
							chunkAction=chunk.id+'__'+tsa[2];
							if(!caUpdated.has(chunkAction)){
								this.updateChunkActionUtility(chunk,tsa[2],ticsSinceActivation,reward);
								caUpdated.add(chunk.id+'__'+tsa[2]);
							}
						}
					}
				}
				//remove outdated episodes
				this.ticStateAction.splice(0,removeSize+1);
			}
			//this.expectedReward=this.actionUtilities[forceAction];
			//add state/action to eligibility trace
			this.ticStateAction.push([this.currentTic-1,this.state,forceAction]);
		}
		//act
		this.log({state:this.state.map(c=>c.id)})
		this.log({action:myAction});
		return myAction;
	}
	modelTrace(cases,independent){
		// cases is an array, where each element is a tuple: [perceptualCues,outputs],
		//	where outputs is an array of [forcedAction,reward] tuples
		//	Ex: cases = [
		//					[ ['gold','circle','large'], [ ['eat', -1], ['pocket', 1] ] ],
		//					[ ['gold','circle','small'], [ ['eat', -.7], ['pocket', .5] ] ],
		//					[ ['red','circle','small'], [ ['eat', .5], ['pocket', -.1] ] ]
		//				]
		var predictions=[],p,i;
		for(i=0;i<cases.length;i++){
			p=[];
			//todo: push only 1st prediction
			for(var actionReward of cases[i][1]){
				p.push(this.step(cases[i][0],0,actionReward[0],true));
				this.step([],actionReward[1]);
				if(independent)this.wake();
			}
			predictions.push(p);
		}
		return predictions;
	}
}


if(typeof(window)==='undefined'){
	/////////////////////////////////////////////////////////////
	// exit handler
	function exitHandler(options,err){
		var agent;
		while(agent=agents.pop())
			agents[i].end();
		if(err)console.error(err.stack);
		if(options.exit)process.exit();
	}
	//starts handler on exit
	process.on('exit', exitHandler.bind(null));
	//catch ctrl+c event
	process.on('SIGINT', exitHandler.bind(null,{exit:true}));
	/////////////////////////////////////////////////////////////



	var a=new CRRL(['left','right'],{},'try',1);

	for(var i=0;i<1000;i++){
		a.step(['a','b','c','d'],0)
		a.step(['g'],1)
	}

	console.log('done');

	a.end();
}
