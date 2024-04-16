// CRRL (Conservative Rational perceptual chunking + Reinforcement Learning)


//TODO:
//	numeric inputs:
//		problem: if num2sym is done outside the model, model will make chunks from same input
//		can be done bitwise based on precision param
//			greedy, with binSize=2 overlapping by 1, mrt multiplier is 2:
//				e.g.: if precision is 6 bits, 10 input symbols per num
//					30 = 29-30, 30-31, 28-31, 30-33, 24-31, 28-35, 16-31, 24-39, 0-31, 16-47
//			or with a multiplier params
//				e.g.: if precision is 6 bits and multiplier is every 2nd bit,
//					all nums covert to 0-63, with 3 symbols per num:
//						63 = 63, 60-63, 48-63
//						7 = 7, 4-7, 0-15
//			possible that you don't need overlap at precise-most level?
//	allow input pre-processing to make all chunks (and process numeric inputs) prior to RL learning
//	allow a non-continuous version (no recency, just frequency)
//		perhaps two diff versions made for repeating and non-repeating learning sets,
//			OR just enable a mechanism to repeat a learning set:
//	expectedReward can do funny things (currently commented out, but could be bio-realistic)
//		actionUtilities for cobined chunks turns negative even tho each component is positive
//			{"reward":1.37,"action":"left","chunk":"b,d","before":0.016,"after":-0.009}
//			{"reward":1.37,"action":"left","chunk":"c,a","before":0.077,"after":0.052}
//			{"reward":1.161,"action":"left","chunk":"c,d,a,b","before":0.062,"after":0.055}
//	ugm -> <0 >0 should instead be a separate discrete param for reward default null/0
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


//Array.prototype.last=function(){return this[this.length-1]};
function last(a){return a[a.length-1]};
function shuffle(a,i,r,t){i=a.length;while(i)r=Math.random()*i--|0,t=a[i],a[i]=a[r],a[r]=t}
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
			last(this.parent1.activationTimes)==currentTic &&
			last(this.parent2.activationTimes)==currentTic);
	}
}
class CRRL{
	constructor(allowedActions=[],params={},name='a',verbose=0){
		// agents.push(this);
		this.VERSION='CRRL1.1';
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
		if(chunk.activationTimes.length==0||last(chunk.activationTimes)!=this.currentTic){ //ensure one activation per tic
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
	learnChunking(){
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
	}
	learnReward(action,reward){
		this.ticStateAction.push([this.currentTic,this.state,action]);
		//propagate reward
		if(this.params.ugm>=0||reward!=0){
			var tsa,caUpdated=new Set(),chunkAction,removeSize=-1,ticsSinceActivation,chunk;
			if(this.params.ugm>0)reward+=this.params.ugm*(this.actionUtilities[forceAction]||0);
			for(var i=this.ticStateAction.length;i-->0;){
				tsa=this.ticStateAction[i];
				ticsSinceActivation=this.currentTic-tsa[0];
				if(ticsSinceActivation>=this.activationDecay.length){
					removeSize=i;
				}else{
					for(var j=0;j<tsa[1].length;j++){
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
			this.learnChunking();
			this.learnReward(forceAction,reward);
		}
		//act
		this.log({state:this.state.map(c=>c.id),action:myAction});
		return myAction;
	}
	modelTrace(cases,independent,learnAllOutputs){
		// cases is an array, where each element is a tuple: [perceptualCues,outputs],
		//	where outputs is an array of [forcedAction,reward] tuples
		//	Ex: cases = [
		//					[ ['gold','circle','large'], [ ['eat', -1], ['pocket', 1] ] ],
		//					[ ['gold','circle','small'], [ ['eat', -.7], ['pocket', .5] ] ],
		//					[ ['red','circle','small'], [ ['eat', .5], ['pocket', -.1] ] ]
		//				]
		var predictions=[],i,j,inputs,actionRewards,reward,forceAction,curLearning=this.learning;
		this.learning=false;
		for(i=0;i<cases.length;i++){
			[inputs,actionRewards]=cases[i];
			predictions.push(this.step(inputs));
			this.learnChunking();
			if(learnAllOutputs){
				learnAllOutputs=new Set(this.allowedActions);
				for(j=0;j<actionRewards.length;j++)
					learnAllOutputs.delete(actionRewards[j][0]);
				for(forceAction of learnAllOutputs){
					this.ticStateAction.pop(); //TODO: this seems to be inappropriately placed? should only work for j>0?
					this.learnReward(forceAction,0);
				}
			}
			for(j=0;j<actionRewards.length;j++){
				[forceAction,reward]=actionRewards[j];
				this.ticStateAction.pop(); //TODO: this seems to be inappropriately placed? should only work for j>0?
				this.learnReward(forceAction,reward);
			}
			if(independent)this.wake();
		}
		this.learning=curLearning;
		return predictions;
	}
	modelTraceBatch(cases,epochs=5,batchSize=50,learnAllOutputs){
		var predictions=[],caseBatch;
		for(var i=0;i<cases.length;i+=batchSize){
			console.log('crrl predicting/learning cases '+i+' to '+(i+batchSize));
			predictions=predictions.concat(this.modelTrace(cases.slice(i,i+batchSize),true,learnAllOutputs));
			caseBatch=Array.from(cases.slice(i,i+batchSize));
			for(var e=1;e<epochs;e++){
				shuffle(caseBatch);
				this.modelTrace(caseBatch,true,learnAllOutputs);
			}
		}
		return predictions;
	}
	predict(inputs){
		var predictions=[],curLearning=this.learning;
		this.learning=false;
		for(var i=0;i<inputs.length;i++){
			predictions.push(this.step(inputs[i]));
		}
		this.learning=curLearning;
		return predictions;
	}
}


if(typeof(window)==='undefined'){
	// /////////////////////////////////////////////////////////////
	// // exit handler
	// function exitHandler(options,err){
		// var agent;
		// while(agent=agents.pop())
			// agents[i].end();
		// if(err)console.error(err.stack);
		// if(options.exit)process.exit();
	// }
	// //starts handler on exit
	// process.on('exit', exitHandler.bind(null));
	// //catch ctrl+c event
	// process.on('SIGINT', exitHandler.bind(null,{exit:true}));
	// /////////////////////////////////////////////////////////////

	try{
		exports.CRRL=CRRL;
	}catch(e){}

	// var a=new CRRL(['left','right'],{},'try',1);

	// for(var i=0;i<1000;i++){
		// a.step(['a','b','c','d'],0)
		// a.step(['g'],1)
	// }

	// console.log('done');

	// a.end();
}
